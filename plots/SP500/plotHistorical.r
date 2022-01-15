# Plot - Historical Data
#
# Color Scheme:
#   Blue: #00529e
#   Red:  #b51b2d
#   Gray: #969696

library(ggplot2)
library(reshape2)
library(grid)
setwd('/Users/troy_sky/Documents/R/Thesis/SP500')
vplayout = function(x, y) viewport(layout.pos.row = x, layout.pos.col = y)

# Data Loading
dataFullIndex = read.csv('dataHistoricalIndex.csv')
dataRetIn = read.csv('dataInSample.csv')

# Figure 1 - Index line plot
histFig1 = ggplot(data = dataFullIndex,aes(x = as.Date(time),y = close,group = 0)) + 
  geom_line(aes(color = '1'),size = 0.5) + 
  scale_color_manual('',labels = 'S&P500 Historical Trend',values = '#00529e') + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.163,0.922)) + 
  theme(legend.key = element_rect(fill = '#00529e',color = NULL)) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = 'Index',title = '')
print(histFig1)
ggsave('histFig1.pdf',histFig1,width = 10,height = 4)

# Figure 2 - Return scatter plot
histFig2 = ggplot(data = dataRetIn,aes(x = ret,y = std)) + 
  geom_point(aes(color = label),size = 2) + 
  xlim(c(-0.05,0.05)) + ylim(c(0,0.04)) + 
  scale_color_manual('States defined in K-Means clustering',
                     values = c('#00529e','#b51b2d','#969696')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.182,0.8)) + 
  theme(legend.key = element_rect(fill = NULL,color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = 'Daily Log return',y = 'Daily volatility',title = '')
print(histFig2)
ggsave('histFig2.pdf',histFig2,width = 10,height = 4)

# Figure 3 - Index line plot & Return bar plot
dataIndexReturn = merge(dataFullIndex,dataRetIn,by = 'time',all.y = T)
histFig3_Up = ggplot(data = dataIndexReturn,aes(x = as.Date(time),y = close,group = 0)) + 
  geom_line(aes(color = '1'),size = 0.5) + 
  scale_color_manual('',labels = 'CSI300 Historical Trend',values = '#00529e') + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.title.y = element_text(vjust = 1.3)) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = 'none') + 
  labs(x = '',y = 'Index (in-sample history)',title = '') + 
  theme(plot.margin = unit(c(0.5,0.5,-0.5,0.5),'cm'))
print(histFig3_Up)

histFig3_Down = ggplot(data = dataIndexReturn,aes(x = as.Date(time),y = ret)) + 
  geom_bar(stat = 'identity',aes(fill = label),position = 'dodge') + 
  scale_fill_manual('',labels = c('bear        ','bull        ','intermediate'),
                    values = c('#00529e','#b51b2d','#969696')) + 
  ylim(c(-0.05,0.05)) + theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.line.x = element_blank()) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(axis.text.x = element_blank()) + 
  theme(axis.text.y = element_text(size = 9)) + 
  theme(legend.position = c(0.5,-0.05)) + 
  theme(legend.direction = 'horizontal') + 
  theme(legend.key = element_rect(color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = 'Daily Log return',title = '') + 
  theme(plot.margin = unit(c(-0.8,0.5,0.5,0.3),'cm'))
print(histFig3_Down)

pdf('histFig3.pdf', width = 10, height = 6)
grid.newpage()
pushViewport(viewport(layout = grid.layout(5, 1)))
print(histFig3_Up, vp = vplayout(1:3, 1))
print(histFig3_Down, vp = vplayout(4:5, 1))
dev.off()

# Figure 4 - Return density
histFig4 = ggplot(data = dataRetIn,aes(x = ret)) + 
  geom_density(color = '#00529e',fill = '#00529e60') + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  labs(x = 'Daily Log return',y = 'Density',title = '')
print(histFig4)
ggsave('histFig4.pdf',histFig4,width = 10,height = 4)