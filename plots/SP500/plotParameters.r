# Plot - HMM Parameters
#
# Color Scheme:
#   Blue: #00529e
#   Red:  #b51b2d
#   Gray: #969696

library(ggplot2)
library(dplyr)
library(reshape2)
library(grid)
setwd('/Users/troy_sky/Documents/R/Thesis/SP500')
vplayout = function(x, y) viewport(layout.pos.row = x, layout.pos.col = y)

# Data Loading
dataParamTemp = read.csv('dataMatParam.csv')
dataParamTemp = dataParamTemp[floor(nrow(dataParamTemp)*2/3):nrow(dataParamTemp),]
dataMu = dataParamTemp %>% select(time,bear = mu1,intermediate = mu2,bull = mu3)
dataSigma = dataParamTemp %>% select(time,bear = sigma1,intermediate = sigma2,bull = sigma3)
dataMu = melt(dataMu,id.vars = 'time')
dataSigma = melt(dataSigma,id.vars = 'time')
dataParam = merge(dataMu,dataSigma,by = c('time','variable')) %>% 
  select(time,label = variable,mu = value.x,sigma = value.y)

# Figure 1 - Parameters scatter plot
paramFig1 = ggplot(data = dataParam,aes(x = mu,y = sigma)) + 
  geom_point(aes(color = label),size = 2) + 
  xlim(c(-0.02,0.02)) + ylim(c(0,0.05)) + 
  scale_color_manual('',values = c('#00529e','#969696','#b51b2d')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.8,0.88)) + 
  theme(legend.key = element_rect(fill = NULL,color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = expression(paste(mu,' of normal distribution')),
       y = expression(paste(sigma,' of normal distribution')))
print(paramFig1)
ggsave('paramFig1.pdf',paramFig1,width = 10,height = 4.3)

# Figure 2 - Parameters density plot
paramFig2Mu = ggplot(data = dataParam,aes(x = mu)) + 
  geom_density(aes(color = label,fill = label)) + 
  xlim(c(-0.02,0.02)) +
  scale_color_manual('States',values = c('#00529e','#969696','#b51b2d')) + 
  scale_fill_manual('States',values = c('#00529e60','#96969660','#b51b2d60')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  labs(x = expression(paste(mu,' of normal distribution')),y = 'Density',title = '') + 
  theme(plot.margin = unit(c(0.5,0.5,0.1,0.5),'cm'))
print(paramFig2Mu)

paramFig2Sigma = ggplot(data = dataParam,aes(x = sigma)) + 
  geom_density(aes(color = label,fill = label)) + 
  xlim(c(0,0.05)) + 
  scale_color_manual('States',values = c('#00529e','#969696','#b51b2d')) + 
  scale_fill_manual('States',values = c('#00529e60','#96969660','#b51b2d60')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  labs(x = expression(paste(sigma,' of normal distribution')),y = 'Density',title = '') + 
  theme(plot.margin = unit(c(0.1,0.5,0.5,0.5),'cm'))
print(paramFig2Sigma)

pdf('paramFig2.pdf', width = 10, height = 6)
grid.newpage()
pushViewport(viewport(layout = grid.layout(2, 1)))
print(paramFig2Mu, vp = vplayout(1, 1))
print(paramFig2Sigma, vp = vplayout(2, 1))
dev.off()
