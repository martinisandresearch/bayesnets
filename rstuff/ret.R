library(tidyverse)
library(gganimate)
library(gifski)
library(transformr)
library(skimr)
library(rjson)
library(reticulate)

use_condaenv("torchenv")
source_python("../experiments/goofy/basic_hive.py")

# long_df <- get_long_results(
#   width_list = list(1L,3L, 10L),
#   momentum_list = list(0.7, 0.95),
#   lr_list = list(0.02, 0.002),
#   num_epochs = 100L,
#   num_bees = 12L,
#   seed = 9L)
# 
# long_df$width %>% unique()
# 
# myact <- long_df %>% 
#   pull(x) %>% 
#   unique() %>% 
#   tibble()
# colnames(myact) <- 'x'
# myact <- myact %>% 
#   mutate(x = x %>% as.numeric(),
#          sin = x %>% sin(),
#          tanh = x %>% tanh(),
#          xtanh = x - (x %>% tanh()),
#          relu = case_when(x < 0 ~ 0,
#                           TRUE ~ x))
# 
# myact %>% 
#   gather(key = 'key', value = 'value', -x) %>% 
#   ggplot(aes(x = x, y = value, col = key))+
#   geom_line()
# 
# 

plot_scaler <- function(obs, obs_range, ax_min, ax_max){
  r <- ax_max - ax_min
  scale <- r/obs_range
  shift <- ax_min
  obs*scale+shift
}


plot_frame = function(long_df, chosen_epoch = 0, facx = "momentum", facy = 'activation', shapevar = 'lr'){
  ymax <- long_df$ypred %>% quantile(0.999) %>% unname()
  ymin <- long_df$ypred %>% quantile(0.001) %>% unname()
  lossmax <- long_df$loss %>% quantile(0.99) %>% unname()
  xmin <- long_df$x %>% min()
  xmax <- long_df$x %>% max()
  num_bees <- long_df$bee %>% unique() %>% length()
  
  plot <- long_df %>% 
    filter(epoch == chosen_epoch) %>% 
    ggplot(aes(x = x, y = ypred, col = as.factor(bee)))+
    geom_point(aes(x = x, y = y), col = 'grey', size = 0.6, alpha = 0.4)+
    geom_line(aes(linetype = as.factor(eval(as.symbol(shapevar)))))+
    geom_point(aes(x = bee %>% plot_scaler(num_bees, xmin, xmax), 
                   y = loss %>% plot_scaler(lossmax, ymin, ymax), 
                   col = as.factor(bee), 
                   shape = as.factor(eval(as.symbol(shapevar)))),
               alpha = 0.7)+
    facet_grid(eval(as.symbol(facy)) ~ eval(as.symbol(facx)))+
    scale_x_continuous(limits = c(xmin, xmax))+
    scale_y_continuous(limits = c(ymin, ymax))+
    scale_color_discrete("Bee")+
    scale_shape(shapevar)+
    scale_linetype(shapevar)
  
  plot
  
}


# long_df %>%
#   filter(width == 3) %>% 
#   plot_frame(99)

animate_gif <- function(long_df, facx = "momentum", facy = 'activation', shapevar = 'lr', title_string = "Swarm Training", epochs_per_second = 20){
  
  ymax <- long_df$ypred %>% quantile(0.999) %>% unname()
  ymin <- long_df$ypred %>% quantile(0.001) %>% unname()
  lossmax <- long_df$loss %>% quantile(0.99) %>% unname()
  xmin <- long_df$x %>% min()
  xmax <- long_df$x %>% max()
  num_bees <- long_df$bee %>% unique() %>% length()
  
  plot <- long_df %>% 
    ggplot(aes(x = x, y = ypred, col = as.factor(bee)))+
    geom_point(aes(x = x, y = y), col = 'grey', size = 0.6, alpha = 0.4)+
    geom_line(aes(linetype = as.factor(eval(as.symbol(shapevar)))))+
    geom_point(aes(x = bee %>% plot_scaler(num_bees, xmin, xmax), 
                   y = loss %>% plot_scaler(lossmax, ymin, ymax), 
                   col = as.factor(bee), 
                   shape = as.factor(eval(as.symbol(shapevar)))),
               alpha = 0.7)+
    facet_grid(eval(as.symbol(facy)) ~ eval(as.symbol(facx)))+
    scale_x_continuous(limits = c(xmin, xmax))+
    scale_y_continuous(limits = c(ymin, ymax))+
    scale_color_discrete("Bee")+
    scale_shape(shapevar)+
    scale_linetype(shapevar)
  
  total_epochs = long_df$epoch %>% max()
  #frames_per_epoch = 2
  #total_frames = total_epochs*frames_per_epoch
  #calculated_duration = total_epochs/epochs_per_second
  
  goodanim <- plot+
    transition_states(epoch,
                      transition_length = 0,
                      state_length = 1)+
    ggtitle(paste0(title_string, " Epoch: {closest_state}"))
  
  animate(goodanim, fps = 20, nframes = total_epochs, width = 1000, end_pause = 10, height = 650, renderer = gifski_renderer())
  
}


#long_df %>% filter(width == 3) %>% animate_gif()

#anim_save("activdemo.gif", path = "../out_animations")

