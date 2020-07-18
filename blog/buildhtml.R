# check the repo has been set otherwise install.packages won't workchooseCRANmirror()
cranrepo = getOption('repos')
if (!startsWith(cranrepo["CRAN"],  "http")){
  cranrepo["CRAN"] <- "https://cran.rstudio.com/"
  print("Setting repo to rstudio")
  options(repos=cranrepo)
}

# get necessary packagelist
pkglist = c('rmarkdown','tidyverse', 'gganimate', 'gifski', 'transformr', 'skimr', 'rjson', 'reticulate')
new.packages <- pkglist[!(pkglist %in% installed.packages()[,"Package"])]
print(new.packages)
if(length(new.packages)) install.packages(new.packages)

# build shit
library(rmarkdown)
rmarkdown::render("baaaasics.Rmd")
