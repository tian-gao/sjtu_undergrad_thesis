# Plot - Number of States
#
# Color Scheme:
#   Blue: #00529e
#   Red:  #b51b2d
#   Gray: #969696

library(ggplot2)
library(reshape2)
library(grid)
setwd('/Users/troy_sky/Documents/R/Thesis/states')

# Data Loading
dataAIC = read.csv('aic.csv')
dataBIC = read.csv('bic.csv')

# Figure 1 - AIC
numStateAIC = ggplot(data = dataAIC,aes(x = as.factor(number),y = value,group = 0)) + 
  geom_line(color = '#00529e',size = 0.5) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  labs(x = 'Number of hidden states',y = 'AIC',title = '')
print(numStateAIC)
ggsave('numStateAIC.pdf',numStateAIC,width = 7,height = 5)

# Figure 2 - BIC
numStateBIC = ggplot(data = dataBIC,aes(x = as.factor(number),y = value,group = 0)) + 
  geom_line(color = '#00529e',size = 0.5) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  labs(x = 'Number of hidden states',y = 'BIC',title = '')
print(numStateBIC)
ggsave('numStateBIC.pdf',numStateBIC,width = 7,height = 5)
