library(tidyverse)
library(gganimate)
library(gifski)
library(transformr)
library(skimr)
library(rjson)
library(reticulate)

use_condaenv("torchenv")
source_python("../experiments/goofy/basic_hive.py")
ypred <- read_csv("../out_sims/ypred.csv")
lossn <- read_csv("../out_sims/loss.csv")

output <- main()

reslist <- output[[1]]
static_params <- reslist[[2]]

adict <- reslist[[1]]
ypred <- adict$results
params <- adict$params
params$hidden

yp <- ypred$ypred

metadata <- fromJSON(file = "../out_sims/metadata.json")

metadata$name
metadata$date


ypred %>% 
  ggplot(aes(x = ypred, y = ypred_val, col = as.factor(swarm)))+
  geom_point()
