# check the repo has been set otherwise install.packages won't workchooseCRANmirror()
cranrepo = getOption('repos')
if (!startsWith(cranrepo["CRAN"],  "http")){
  cranrepo["CRAN"] <- "https://cran.rstudio.com/"
  print("Setting repo to rstudio")
  options(repos=cranrepo)
}

# sudo apt-get update
# suto apt-get install libudunits2-dev

# get necessary packagelist
pkglist = c('rmarkdown','tidyverse', 'gganimate', 'gifski', 'transformr', 'skimr', 'rjson', 'reticulate')
new.packages <- pkglist[!(pkglist %in% installed.packages()[,"Package"])]
print(new.packages)
if(length(new.packages)) install.packages(new.packages)

# build shit
library(rmarkdown)
rmarkdown::render("baaaasics.Rmd", output_dir = "dist", clean=TRUE)
rmarkdown::render("baaaasics.Rmd", output_dir = "dist", clean=TRUE)
