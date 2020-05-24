library(tidyverse)
library(gganimate)
library(gifski)
library(transformr)
library(skimr)
library(rjson)
library(reticulate)

use_condaenv("torchenv")
source_python("../experiments/goofy/basic_hive.py")

#output <- main()
long_df <- get_long_results()

ymax <- long_df$ypred %>% quantile(0.999) %>% unname()
ymin <- long_df$ypred %>% quantile(0.001) %>% unname()
lossmax <- long_df$loss %>% quantile(0.999) %>% unname()
xmin <- long_df$x %>% min()
xmax <- long_df$x %>% max()
num_bees <- long_df$bee %>% unique() %>% length()


plot_scaler <- function(obs, obs_range, ax_min, ax_max){
  r <- ax_max - ax_min
  scale <- obs_range/r
  shift <- ax_min
  obs*scale+shift
}


facx = "momentum"
facy = 'activation'
shapevar = 'lr'

#good_plot <- 

long_df %>% 
  #filter(epoch == 20) %>% 
  ggplot(aes(x = x, y = ypred, col = as.factor(bee)))+
  geom_point(aes(x = x, y = y), col = 'grey', size = 0.6, alpha = 0.4)+
  geom_line(aes(linetype = as.factor(eval(as.symbol(shapevar)))))+
  geom_point(aes(x = bee %>% plot_scaler(num_bees, xmin, xmax), y = loss %>% plot_scaler(lossmax, ymin, ymax), col = as.factor(bee), shape = as.factor(eval(as.symbol(shapevar)))))+
  facet_grid(eval(as.symbol(facy)) ~ eval(as.symbol(facx)))+
  scale_x_continuous(limits = c(xmin, xmax))+
  scale_y_continuous(limits = c(ymin, ymax))+
  scale_color_discrete("Bee")+
  scale_shape(shapevar)+
  scale_linetype(shapevar) -> good_plot

goodanim <- good_plot+
  transition_states(epoch,
                    transition_length = 1,
                    state_length = 1)+
  ggtitle(paste0("Swarm training at epoch {closest_state}"))

animate(goodanim, duration = 5, fps = 20, nframes = 100, width = 1000, height = 650, renderer = gifski_renderer())


anim_save("activdemo.gif", path = "../out_animations")

