library(tidyverse)
library(gganimate)
library(gifski)
library(transformr)
library(skimr)


unique_string = "msmall"

data = read_csv(paste0("../out_sims/sim_", unique_string, "_data.csv"), col_names = F)
loss =  read_csv(paste0("../out_sims/sim_", unique_string, "_loss.csv"), col_names = F)
xy =  read_csv(paste0("../out_sims/sim_", unique_string, "_xy.csv"), col_names = F)
params = read_csv(paste0("../out_sims/sim_", unique_string, "_params.csv"))

colnames(xy) <- c("x", "y", "sim_sig")

colnames(data) <- xy %>% 
  head(length(colnames(data))-2) %>% 
  pull(x) %>% append(c("bee", "sim_sig"))

colnames(loss) <- c("loss", "bee", "sim_sig")

df_n <- data %>% 
  group_by(sim_sig) %>% 
  nest()

xy_n <- xy %>% 
  group_by(sim_sig) %>% 
  nest()
colnames(xy_n) <- c("sim_sig", "xy")

df_n <- df_n %>% 
  left_join(xy_n, by = "sim_sig")

df_n <- df_n %>% 
  left_join(params, by = "sim_sig")

loss_n <- loss %>%
  group_by(sim_sig) %>% 
  nest()
colnames(loss_n) <- c("sim_sig", "loss")

df_n <- df_n %>% 
  left_join(loss_n, by = "sim_sig")

df_n <- df_n %>% 
  mutate(data = data %>% map(~.x %>% group_by(bee) %>% mutate(epoch = row_number()) %>% ungroup())) %>%
  mutate(loss = loss %>% map(~.x %>% group_by(bee) %>% mutate(epoch = row_number()) %>% ungroup())) %>%
  mutate(long_data = data %>% map(~.x %>% gather(key = "x", value = "y", -bee, -epoch))) %>% 
  mutate(long_data = long_data %>% map(~.x %>% mutate(x = as.numeric(x))))


long_data = df_n %>% 
  select(long_data, funcname, hidden,  width, momentum, lr, sim_sig) %>% 
  unnest(long_data) %>% 
  ungroup()

loss = df_n %>% 
  select(loss, funcname, hidden,  width, momentum, lr, sim_sig) %>% 
  unnest(loss) %>% 
  ungroup()

truefunc = long_data %>% 
  select(x) %>% 
  unique() %>% 
  mutate(y = sin(x))

long_data %>% skim()



hidden_layers = 2

#good_plot <- 
loss %>% 
  filter(hidden == hidden_layers) %>% 
  filter(bee < 12) %>%
  filter(epoch ==  1) %>% 
  ggplot(aes(x = bee*6/12-3, y = loss*4-1.5, col = as.factor(bee), shape = as.factor(lr)))+
  geom_point(alpha = 0.8)+
  geom_line(data = long_data %>% filter(bee < 12) %>% filter(hidden ==hidden_layers) %>% filter(epoch == 1) , aes(x = x, y = y, col = as.factor(bee), linetype = as.factor(lr)), alpha = 0.6, size = 0.7)+
  facet_grid(width~momentum)+
  scale_y_continuous(limits = c(-1.5,1.5))+
  geom_point(data = truefunc, aes(x = x, y = y), col = "black", size = 0.5, shape = 1, alpha = 0.5)


goodanim <- good_plot+
  transition_states(epoch,
                    transition_length = 1,
                    state_length = 1)+
  ggtitle(paste0("Swarm ", hidden_layers, " depth training at epoch {closest_state}"))


animate(goodanim, duration = 16, fps = 25, nframes = 400, width = 1000, height = 650, renderer = gifski_renderer())
anim_save("msmalldemo.gif", path = "../out_animations")

### animations ----
  

sample_data = df_n$data[[1]]

sample_data_l = sample_data %>% 
  gather(key = "x", value = "y", -bee, -epoch)
sample_data_l =  sample_data_l %>% 
  mutate(x  = x %>% as.numeric())

plot <- sample_data_l %>% 
  filter(epoch < 80) %>% 
  filter(bee < 12) %>% 
  ggplot(aes(x = x, y = y, col = as.factor(bee)))+
  geom_line()

anim2 <- plot + 
  transition_states(epoch,
                    transition_length = 2,
                    state_length = 1)

animate(anim2, duration = 10, fps = 20, width = 500, height = 400, renderer = gifski_renderer())
anim_save("test.gif", path = "../out_animations")
