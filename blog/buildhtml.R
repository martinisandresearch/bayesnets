# build shit

rmarkdown::render("basics.Rmd", output_dir = "dist", clean=TRUE)
rmarkdown::render("lrmom.Rmd", output_dir = "dist", clean=TRUE)
