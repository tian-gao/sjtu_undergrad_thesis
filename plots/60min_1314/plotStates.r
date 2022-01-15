# Plot - HMM States
#
# Color Scheme:
#   Blue: #00529e
#   Red:  #b51b2d
#   Gray: #969696

library(ggplot2)
library(dplyr)
library(reshape2)
library(grid)
setwd('/Users/troy_sky/Documents/R/Thesis/60min_1314')
vplayout = function(x, y) viewport(layout.pos.row = x, layout.pos.col = y)

# Data Loading
dataRetStates = read.csv('dataHMMStates.csv')
dataFullIndex = read.csv('dataHistoricalIndex.csv')
dataRetIn = read.csv('dataInSample.csv')
dataIndexStates = merge(dataRetStates,dataFullIndex,by = 'time',all.x = T)

# Figure 1 - Index line & States bar plot
ind = c(57,217,377,541,697,869)
statesFig1 = ggplot(data = dataIndexStates,aes(x = time,y = close,group = 1)) + 
  geom_line(color = '#969696') + 
  geom_bar(stat = 'identity',position = 'dodge',aes(fill = label),width = 1) + 
  scale_fill_manual('Hidden States',values = c('#00529e60','#b51b2d60','#96969660')) + 
  scale_x_discrete(breaks = dataIndexStates$time[ind],labels = substr(dataIndexStates$time[ind],1,7)) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  labs(x = '',y = 'CSI300 historical trend & hidden states')
print(statesFig1)
ggsave('statesFig1.pdf',statesFig1,width = 10,height = 4)

# Figure 2 - States bar plot
statesFig2 = ggplot(data = dataIndexStates,aes(x = as.Date(time),y = close)) + 
  geom_bar(stat = 'identity',aes(fill = label)) + 
  facet_grid(label ~.) + 
  scale_fill_manual('Hidden States',values = c('#00529e60','#b51b2d60','#96969660')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_rect(size = 0.5,color = '#969696')) + 
  theme(axis.ticks.x = element_blank()) + 
  labs(x = '',y = 'CSI300 historical hidden states')
print(statesFig2)
ggsave('statesFig2.pdf',statesFig2,width = 10,height = 4)

# Figure 3 - Return scatter plots
dataRetInComp = merge(dataRetStates,dataRetIn,by = c('time','ret'),all.y = T) %>%
  select(time,ret,std,labelHMM = label.x,labelKMeans = label.y) %>% 
  mutate(labelDiff = labelHMM == labelKMeans)

statesFig3_HMM = ggplot(data = dataRetInComp,aes(x = ret,y = std)) + 
  geom_point(aes(color = labelHMM),size = 2) + 
  xlim(c(-0.05,0.05)) + ylim(c(0,0.04)) + 
  scale_color_manual('States',values = c('#00529e','#b51b2d','#969696')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.182,0.83)) + 
  theme(legend.key = element_rect(fill = NULL,color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = 'HMM implied states',y = '',title = '')
print(statesFig3_HMM)

statesFig3_KMeans = ggplot(data = dataRetInComp,aes(x = ret,y = std)) + 
  geom_point(aes(color = labelKMeans),size = 2) + 
  xlim(c(-0.05,0.05)) + ylim(c(0,0.04)) + 
  scale_color_manual('States',values = c('#00529e','#b51b2d','#969696')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.182,0.83)) + 
  theme(legend.key = element_rect(fill = NULL,color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = 'K-Means implied states',y = '',title = '')
print(statesFig3_KMeans)

statesFig3_Diff = ggplot(data = dataRetInComp,aes(x = ret,y = std)) + 
  geom_point(aes(color = labelDiff),size = 2) + 
  xlim(c(-0.05,0.05)) + ylim(c(0,0.04)) + 
  scale_color_manual('States',
                     labels = c('different','same'),
                     values = c('#b51b2d','#969696')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.15,0.857)) + 
  theme(legend.key = element_rect(fill = NULL,color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = 'Differences in implied states of two models',y = '',title = '')
print(statesFig3_Diff)

pdf('statesFig3.pdf', width = 15, height = 5)
grid.newpage()
pushViewport(viewport(layout = grid.layout(1, 3)))
print(statesFig3_HMM, vp = vplayout(1, 1))
print(statesFig3_KMeans, vp = vplayout(1, 2))
print(statesFig3_Diff, vp = vplayout(1, 3))
dev.off()
