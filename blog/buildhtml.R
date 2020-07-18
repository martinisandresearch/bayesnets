# check the repo has been set otherwise install.packages won't work
cranrepo = getOption('repos')
if (!startsWith(cranrepo["CRAN"],  "http")){
  cranrepo["CRAN"] <- "https://cran.rstudio.com/"
  print("Setting repo to rstudio")
  options(repos=cranrepo)
}

install.packages(c('rmarkdown','tidyverse', 'gganimate', 'gifski', 'transformr', 'skimr', 'rjson', 'reticulate'))
library(rmarkdown)
