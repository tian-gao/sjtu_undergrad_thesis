# Plot - HMM Transition Matrix
#
# Color Scheme:
#   Blue: #00529e
#   Red:  #b51b2d
#   Gray: #969696

library(ggplot2)
library(dplyr)
library(reshape2)
library(grid)
library(circlize)
setwd('/Users/troy_sky/Documents/R/Thesis/SP500')
vplayout = function(x, y) viewport(layout.pos.row = x, layout.pos.col = y)

# Data Loading
dataTransTemp = read.csv('dataMatTrans.csv')
dataTransTemp = dataTransTemp[ceiling(nrow(dataTransTemp)*2/3):nrow(dataTransTemp),]

# Figure 1 - Transition matrix chord plot
states = c('bear','intermediate','bull')
dataTransTypical = data.frame(from = rep(states,each = 3),
                              to = rep(states,times = 3),
                              value = as.numeric(t(dataTransTemp[80,2:10])))

pdf('transFig1.pdf', width = 10, height = 2.5)
par(mar = c(0,0,0,0),mfrow = c(1,4))
chordDiagramFromDataFrame(dataTransTypical,link.border = '#969696',
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529ecc',intermediate = '#969696cc',bull = '#b51b2dcc'))
chordDiagramFromDataFrame(dataTransTypical,
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529e10',intermediate = '#96969610',bull = '#b51b2dcc'))
chordDiagramFromDataFrame(dataTransTypical,
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529e10',intermediate = '#969696cc',bull = '#b51b2d10'))
chordDiagramFromDataFrame(dataTransTypical,
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529ecc',intermediate = '#96969610',bull = '#b51b2d10'))
dev.off()

# Figure 2 - Transition matrix change line plot
dataTrans = dataTransTemp
colnames(dataTrans) = c('time',paste(rep(c('BE.','IN. ','BU.'),each = 3),
                                     rep(c('BE.','IN.','BU.'),times = 3),sep = ' -> '))
dataTrans = melt(dataTrans,id.vars = 'time')

transFig2 = ggplot(data = dataTrans,aes(x = as.Date(time),y = value,group = variable)) + 
  geom_line(aes(color = variable),size = 0.5) + 
  scale_color_manual('Transition',
                     values = c('#00529e','#00929e','#80529e',
                                '#323232','#969696','#dcdcdc',
                                '#b51b2d','#b59b6d','#f51b2d')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.key = element_rect(fill = NULL,color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = '')
print(transFig2)
ggsave('transFig2.pdf',transFig2,width = 10,height = 3.5)

# Figure 3 - Transition matrix change cord plots
ind = dataTransTemp$time %in% c('2015-04-01','2015-04-29','2015-06-29',
                                '2015-07-07','2015-07-29','2016-03-03')
dataTransNodes = data.frame(from = dataTransTypical$from,to = dataTransTypical$to,
                            matrix(as.numeric(t(dataTransTemp[ind,2:10])),nrow = 9,ncol = 6))

pdf('transFig3.pdf', width = 9, height = 6)
par(mar = c(0,1,3,1),mfrow = c(2,3))
chordDiagramFromDataFrame(dataTransNodes[,c(1,2,3)],
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529ebb',intermediate = '#969696bb',bull = '#b51b2dbb'))
title(dataTransTemp$time[ind][1],cex.main = 1.2,font.main = 1)
chordDiagramFromDataFrame(dataTransNodes[,c(1,2,4)],
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529ebb',intermediate = '#969696bb',bull = '#b51b2dbb'))
title(dataTransTemp$time[ind][2],cex.main = 1.2,font.main = 1)
chordDiagramFromDataFrame(dataTransNodes[,c(1,2,5)],
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529ebb',intermediate = '#969696bb',bull = '#b51b2dbb'))
title(dataTransTemp$time[ind][3],cex.main = 1.2,font.main = 1)
chordDiagramFromDataFrame(dataTransNodes[,c(1,2,6)],
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529ebb',intermediate = '#969696bb',bull = '#b51b2dbb'))
title(dataTransTemp$time[ind][4],cex.main = 1.2,font.main = 1)
chordDiagramFromDataFrame(dataTransNodes[,c(1,2,7)],
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529ebb',intermediate = '#969696bb',bull = '#b51b2dbb'))
title(dataTransTemp$time[ind][5],cex.main = 1.2,font.main = 1)
chordDiagramFromDataFrame(dataTransNodes[,c(1,2,8)],
                          annotationTrack = c('name','grid'),
                          annotationTrackHeight = c(0.1,0.02),
                          grid.col = c(bear = '#00529ebb',intermediate = '#969696bb',bull = '#b51b2dbb'))
title(main = dataTransTemp$time[ind][6],cex.main = 1.2,font.main = 1)
dev.off()
