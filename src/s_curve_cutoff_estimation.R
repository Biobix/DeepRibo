library("splines")
library("SiZer")
get_cutoff_values <- function(path,dest){
  print('Fitting model parameters...')
  # Loading data
  df <- read.csv(path)
  df <- df[df$label==1,c("rpk_elo","coverage_elo")]
  # Transforming data
  mask <- df$rpk_elo != 0
  df <- df[mask,]
  df$rpk_elo <- log(df$rpk_elo)
  rownames(df) <- NULL
  # Fitting the S-curve
  formula <- coverage_elo ~ SSfpl(rpk_elo, D, A, C, B)
  model <- nls(formula, data=df)
  # Prepare data
  df_fit <- cbind(df[order(df$rpk_elo),])
  fit_idx <- length(df$rpk_elo)
  decr_step <- max(round(fit_idx/800),4)
  MINCOV <- 1

  while (MINCOV>0.60){
    df_temp <- cbind(df_fit[1:fit_idx,])
    # Fit bent curve
    bent_curve <- bent.cable(df_temp$rpk_elo, 
                            predict(model, df_temp))
    MINRPKM <- round(bent_curve$alpha, digits = 6)
    MINCOV <- round(predict(bent_curve, MINRPKM), digits = 6)
    # Drop upper right data points in order to force 
    # the function to fit to the lower bent
    if (MINCOV>0.60){
      print('Bent point fitted to upper bent: Refitting model parameters...')
    }
    fit_idx <- fit_idx - decr_step
  }  
  dest <- paste(dest,".png",sep="")
  # Plot data
  png(filename=dest, width=900, height=900)
  plot(df$rpk_elo, df$coverage_elo, xlab="RPKM",ylab="Coverage")
  title("four parameter S curve estimation")
  points(df$rpk_elo, predict(model, df), col="green")
  points(df$rpk_elo, predict(bent_curve, df)[,1], col="red")  
  abline(h=MINCOV, col="blue")
  abline(v=MINRPKM, col="blue")
  text(min(df$rpk_elo)+3, 0.9, paste("min_RPKM= ", exp(MINRPKM),
				     "\nmin_COV= ", MINCOV))
  dev.off()
  return(list("min_RPKM" = exp(MINRPKM), "min_coverage" = MINCOV))
}
