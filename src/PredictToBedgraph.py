#####################################
#  DeepRibo: precise gene annotation of prokaryotes using deep learning
#  and ribosome profiling data
#
#  Copyright (C) 2018 J. Clauwaert, G. Menschaert, W. Waegeman
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  For more (contact) information visit http://www.biobix.be/DeepRibo
#####################################

import sys
import numpy as np
import pandas as pd
import argparse
from argparse import ArgumentDefaultsHelpFormatter as dhf


def predictToBedgraph(df_path, dest_path, count, compare=False):
    df = pd.read_csv(df_path)
    count = np.sum(df['label'])
    df = df.copy()
    df['chrom'] = df['locus'].str.split(':').str[0]
    df['signal'] = np.ones(len(df))
    df['start_site'] = df['start_site']-1
    mask_strand = df['strand'] == '-'
    df.loc[mask_strand, 'start_site'] = df.loc[mask_strand, 'start_site']+1
    df.loc[mask_strand, 'stop_site'] = df.loc[mask_strand, 'stop_site']-3
    df.loc[mask_strand, ['start_site',
                         'stop_site']] = df.loc[mask_strand,
                                                ['stop_site',
                                                 'start_site']].values
    df['temp'] = df['start_site']+3
    df = df.sort_values(['chrom', 'start_site'])
    mask_false = np.logical_and(df["label"] == False,
                                df["SS_pred_rank"] < count)
    mask_correct = np.logical_and(df["label"],
                                  df["SS_pred_rank"] < count)
    mask_pos_set = df["label"]
    mask_pred = df['SS_pred_rank'] < count
    mask_multiple = df['pred_rank'] < count
    mask_all = np.full(len(df), True)

    if compare:
        idx = [0, 1, 2, 3, 4, 5, 6]
    else:
        idx = [5, 6, 7, 8]

    files = np.array(['_fpG.bedgraph', '_tpG.bedgraph', '_fpT.bedgraph',
                      '_tpT.bedgraph', '_pos.bedgraph', '_all.bedgraph',
                      '_mmT.bedgraph', '_G.bedgraph', '_T.bedgraph'])

    colors = np.array(["255,0,0", "124,252,0", "255,0,0", "124,252,0",
                       "0,0,252", "124,124,124", "0,0,252", "124,252,0",
                       "124,252,0"])

    masks = np.array([mask_false, mask_correct, mask_false, mask_correct,
                      mask_pos_set, mask_all, mask_multiple, mask_pred,
                      mask_pred])

    for metadata in zip(files[idx], colors[idx], masks[idx]):
        with open('{}{}'.format(dest_path, metadata[0]), 'w') as f:
            f.write("track type=bedGraph name='BedGraph Format'"
                    "description='BedGraph format' visibility=full"
                    "color={}\n".format(metadata[1]))
            if metadata[0] not in ['_fpG.bedgraph', '_tpG.bedgraph',
                                   '_G.bedgraph']:
                temp = df.copy()
                temp.loc[mask_strand,
                         'start_site'] = df.loc[mask_strand,
                                                ['stop_site']].values-3
                temp.loc[mask_strand, 'temp'] = df.loc[mask_strand,
                                                       ['stop_site']].values
                temp.loc[metadata[2], ['chrom', 'start_site',
                                       'temp', 'signal']].to_csv(f, index=None,
                                                                 header=None,
                                                                 sep="\t")
            else:
                temp = df.loc[metadata[2], ['chrom', 'start_site',
                                            'stop_site', 'signal']]
                temp.to_csv(f, index=None, header=None, sep='\t')
            f.truncate()
            f.close()


def main():
    parser = argparse.ArgumentParser(description="Create .bedgraph files "
                                     "of top k ranked predictions made by "
                                     "DeepRibo", formatter_class=dhf)
    parser.add_argument('csv_path', type=str, help="Path to csv containing "
                        "predictions")
    parser.add_argument('dest_path', type=str, help="Path to destination, as "
                        "multiple files are created, no file extension should "
                        "be included")
    parser.add_argument('k', type=str, help="Visualize the top k ranked "
                        "predictions")
    parser.add_argument('--compare', action='store_true', help="compare "
                        "predictions with annotated labels, visualizes "
                        "distinction between predictions in agreement and "
                        "disagreement. (only possible if --gtf flag was used"
                        " when parsing dataset)")
    args = parser.parse_args()
    predictToBedgraph(args.csv_path, args.dest_path, args.k, args.compare)


if __name__ == "__main__":
    main()
