library("splines")
library("SiZer")
get_cutoff_values <- function(path,dest){
  df <- read.csv(path)
  df<- df[df$label==1,c("rpk_elo","coverage_elo")]
  mask <- df$rpk_elo != 0
  df <- df[mask,]
  df$rpk_elo <- log(df$rpk_elo)
  formula <- coverage_elo ~ SSfpl(rpk_elo, D, A, C, B)
  model <- nls(formula, data=df)
  bent_curve<- bent.cable(df$rpk_elo, predict(model, df$rpk_elo))
  
  png(filename=paste(path,".png",sep=""))
  plot(df$rpk_elo, df$coverage, xlab="RPKM",ylab="Coverage")
  title("four parameter S curve estimation")
  points(predict(model, df$rpk_elo)~ df$rpk_elo, col="green")
  points( df$rpk_elo, predict(bent_curve, df$rpk_elo), col="red")
  
  MINRPKM <- round(bent_curve$alpha, digits = 2)
  MINCOV <- round(predict(bent_curve, MINRPKM), digits = 2)
  abline(h=MINCOV, col="blue")
  abline(v=MINRPKM, col="blue")
  dev.off()
  return(list("min_RPKM" = exp(MINRPKM), "min_coverage" = MINCOV))
}