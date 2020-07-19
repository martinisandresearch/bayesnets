# build shit

rmarkdown::render("blog/basics.Rmd", output_dir = "blog/dist", clean=TRUE)
rmarkdown::render("blog/lrmom.Rmd", output_dir = "blog/dist", clean=TRUE)
