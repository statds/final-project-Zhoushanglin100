######################################################################################
# Function to compute the bispectrum of a time series x
# sm.c is a constant from Hinich's paper

name <- "Beef"
path <-"/Users/minda/Dropbox/Uconn/6494 Data Science/project_Zhou/"
dyn.load(paste(path, "bispectrum/allfortranmac.so", sep = ''))

bispec <- function(x, sm.c = 0.625){
  #
  xx     <- x - mean(x)
  n      <- length(x)
  m      <- trunc(n^sm.c)
  sx     <- rep(0,m)
  z      <- matrix(0, m, m)
  nsq    <- 0
  bs.out <- .Fortran("sq_bispec",
                     as.double(xx),
                     as.integer(n),
                     as.integer(m),
                     as.integer(m),
                     sx = as.double(sx),
                     z = as.double(z),
                     nsq = as.integer(nsq))
  return(list(sx = bs.out$sx, z = matrix(bs.out$z,m,m), nsq = bs.out$nsq, freqs = seq(0,0.5,length=m)))
}


#############################################################################################

# get Train
train <- read.table(paste(path, name, "/", name, "_TRAIN", sep = ''), sep = ",")
data_trn <- t(train[,2:dim(train)[2]])
label_trn <- train[,1]
write.table(label_trn, paste(path, name, "_bisp/train_label", sep = ''), row.names = FALSE, col.names = FALSE)

n=length(label_trn)
sm.c=0.625   # constant from Hinich paper
m=n^sm.c

for (i in 1:dim(data_trn)[2]){
  ser = data_trn[, i]
  bispects.data_trn <- z <- matrix(0, m, m)
  bispects.data_trn <- bispec(ser,sm.c)$z
  write.table(bispects.data_trn, paste(path, name, "_bisp/train_ser_", i, sep = ''), row.names = FALSE, col.names = FALSE)
}

######################################################################################

# get Test
test <- read.table(paste(path, name, "/", name, "_TEST", sep = ''), sep = ",")
data_tst <- t(test[,2:dim(test)[2]])
label_tst <- test[,1]
write.table(label_tst, paste(path, name, "_bisp/test_label", sep = ''), row.names = FALSE, col.names = FALSE)

n=length(label_tst)
sm.c=0.625   # constant from Hinich paper
m=n^sm.c

for (i in 1:dim(data_tst)[2]){
  ser = data_tst[, i]
  bispects.data_tst <- z <- matrix(0, m, m)
  bispects.data_tst <- bispec(ser,sm.c)$z
  write.table(bispects.data_tst, paste(path, name, "_bisp/test_ser_", i, sep = ''), row.names = FALSE, col.names = FALSE)
}

######################################################################################

# Draw Plot
png(paste(path, name, "_plot/train_ts_ser_1.png", sep = ''))
ts.plot(data_trn[, 1], xlab = "Label 1")
dev.off()
png(paste(path, name, "_plot/train_ts_ser_7.png", sep = ''))
ts.plot(data_trn[, 7], xlab = "label 2")
dev.off()
png(paste(path, name, "_plot/train_ts_ser_13.png", sep = ''))
ts.plot(data_trn[, 13], xlab = "label 3")
dev.off()
