# Plot - HMM Prediction
#
# Color Scheme:
#   Blue: #00529e
#   Red:  #b51b2d
#   Gray: #969696

library(ggplot2)
library(dplyr)
library(reshape2)
library(grid)
setwd('/Users/troy_sky/Documents/R/Thesis/10min_1415')
vplayout = function(x, y) viewport(layout.pos.row = x, layout.pos.col = y)

# Data Loading
dataFullIndex = read.csv('dataHistoricalIndex.csv')
dataRetPrediction = read.csv('dataHMMPrediction.csv')
dataRetIn = read.csv('dataInSample.csv')

numStdDay = 5
numStdMin = 240/10
dataOut = dataRetPrediction[(nrow(dataRetIn)+1):(nrow(dataRetPrediction)-1),]
dataOut$real = diff(log(dataFullIndex$close))[(nrow(dataRetIn)+numStdDay*numStdMin+1):(nrow(dataFullIndex)-1)]
dataOut$right = sign(dataOut$ret) == sign(dataOut$real)
winRatio = sum(dataOut$right) / nrow(dataOut)
print(winRatio)

# Figure 1 - Prediction full index line
dataIndexPredictionFull = merge(dataFullIndex,dataRetPrediction,by = 'time',all.x = T)
dataIndexPredictionFull[numStdDay*numStdMin,c('ret','std')] = 0
dataIndexPredictionFull = na.omit(dataIndexPredictionFull)
dataIndexPredictionFull$cumret = cumsum(dataIndexPredictionFull$ret)
dataIndexPredictionFull$predict = dataIndexPredictionFull$close[1]*exp(dataIndexPredictionFull$cumret)
dataIndexPredictionFullPlot = melt(dataIndexPredictionFull[,c('time','close','predict')],id.vars = 'time')

predictionFig1 = ggplot(data = dataIndexPredictionFullPlot,aes(x = as.Date(time),y = value)) + 
  geom_line(aes(color = variable)) + 
  scale_color_manual('CSI300 historical trend & predicted trend',
                     labels = c('historical','predicted'),
                     values = c('#00529e','#b51b2d')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.25,0.8)) + 
  theme(legend.key = element_rect(color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = '',title = '')
print(predictionFig1)
ggsave('predictionFig1.pdf',predictionFig1,width = 10,height = 4)

# Figure 2 - Prediction partial index line & error bar
dataIndexPrediction = dataIndexPredictionFull[(nrow(dataRetIn)+1):nrow(dataIndexPredictionFull),]
dataIndexPredictionPlot = melt(dataIndexPrediction[,c('time','close','predict')],id.vars = 'time')
dataIndexPredictionError = dataIndexPrediction %>% 
  mutate(dif = predict - close,sign = sign(dif)) %>% 
  select(time,dif,sign)

predictionFig2_Up = ggplot(data = dataIndexPredictionPlot,aes(x = as.Date(time),y = value)) + 
  geom_line(aes(color = variable),size = 0.5) + 
  scale_color_manual('CSI300 historical trend & predicted trend',
                     labels = c('historical','predicted'),
                     values = c('#00529e','#b51b2d')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.22,0.85)) + 
  theme(legend.key = element_rect(color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = '',title = '') + 
  theme(plot.margin = unit(c(0.5,0.5,-0.5,0.5),'cm'))
print(predictionFig2_Up)

predictionFig2_Down = ggplot(data = dataIndexPredictionError,aes(x = as.Date(time),y = abs(dif))) + 
  geom_bar(stat = 'identity',aes(fill = as.factor(sign)),position = 'dodge') + 
  scale_fill_manual('',labels = c('negative','positive'),
                       values = c('#00529e','#b51b2d')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.line.x = element_blank()) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(axis.text.x = element_blank()) + 
  theme(legend.position = c(0.07,0.81)) + 
  theme(legend.direction = 'vertical') + 
  theme(legend.key = element_rect(color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = '',title = '') + 
  theme(plot.margin = unit(c(-0.8,0.5,0.5,0.7),'cm'))
print(predictionFig2_Down)

pdf('predictionFig2.pdf', width = 10, height = 6)
grid.newpage()
pushViewport(viewport(layout = grid.layout(5, 1)))
print(predictionFig2_Up, vp = vplayout(1:3, 1))
print(predictionFig2_Down, vp = vplayout(4:5, 1))
dev.off()

# Figure 3 - Prediction line plot with std
dataIndexPrediction[1,c('ret','std')] = 0
dataIndexPrediction$cumret = cumsum(dataIndexPrediction$ret)
dataIndexPredictionStd = dataIndexPrediction %>% 
  mutate(predict_n1 = close[1]*exp(cumret - std),
         predict_n2 = close[1]*exp(cumret - 2*std),
         predict_p1 = close[1]*exp(cumret + std),
         predict_p2 = close[1]*exp(cumret + 2*std)) %>% 
  select(time,close,predict_n2,predict_n1,predict,predict_p1,predict_p2)
dataIndexPredictionStdPlot = melt(dataIndexPredictionStd,id.vars = 'time')

predictionFig3 = ggplot(data = dataIndexPredictionStdPlot,aes(x = as.Date(time),y = value)) + 
  geom_line(aes(color = variable,linetype = variable)) + 
  scale_color_manual('',
                     labels = c('historical','predicted - 2 std.','predicted - 1 std.',
                                'predicted','predicted + 1 std.','predicted + 2 std.'),
                     values = c('#00529e','#969696','#b51b2d','#b51b2d','#b51b2d','#969696')) + 
  scale_linetype_manual('',
                        labels = c('historical','predicted - 2 std.','predicted - 1 std.',
                                   'predicted','predicted + 1 std.','predicted + 2 std.'),
                        values = c(1,4,2,1,2,4)) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.key = element_rect(color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = '',title = '')
print(predictionFig3)
ggsave('predictionFig3.pdf',predictionFig3,width = 12,height = 4)

# Figure 4 - Prediction partial index line & error bar
dataIndexPredictionDynamic = dataIndexPrediction %>% 
  mutate(predictDyn = c(close[1],close[1:(nrow(dataIndexPrediction)-1)]*exp(ret[2:nrow(dataIndexPrediction)]))) %>% 
  select(time,close,predict = predictDyn)
dataIndexPredictionDynamicPlot = melt(dataIndexPredictionDynamic[,c('time','close','predict')],id.vars = 'time')
dataIndexPredictionDynamicError = dataIndexPredictionDynamic %>% 
  mutate(dif = predict - close,sign = sign(dif)) %>% 
  select(time,dif,sign)

predictionFig4_Up = ggplot(data = dataIndexPredictionDynamicPlot,aes(x = as.Date(time),y = value)) + 
  geom_line(aes(color = variable),size = 0.5) + 
  scale_color_manual('CSI300 historical trend & predicted trend',
                     labels = c('historical','predicted'),
                     values = c('#00529e','#b51b2d')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.position = c(0.8,0.3)) + 
  theme(legend.key = element_rect(color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = '',title = '') + 
  theme(plot.margin = unit(c(0.5,0.5,-0.5,0.1),'cm'))
print(predictionFig4_Up)

predictionFig4_Down = ggplot(data = dataIndexPredictionDynamicError,aes(x = as.Date(time),y = abs(dif))) + 
  geom_bar(stat = 'identity',aes(fill = as.factor(sign)),position = 'dodge') + 
  scale_fill_manual('',labels = c('negative','zero','positive'),
                    values = c('#00529e','#969696','#b51b2d')) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.line.x = element_blank()) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(axis.text.x = element_blank()) + 
  theme(legend.position = c(0.07,0.75)) + 
  theme(legend.direction = 'vertical') + 
  theme(legend.key = element_rect(color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = '',title = '') + 
  theme(plot.margin = unit(c(-0.8,0.5,0.5,0.3),'cm'))
print(predictionFig4_Down)

pdf('predictionFig4.pdf', width = 10, height = 6)
grid.newpage()
pushViewport(viewport(layout = grid.layout(5, 1)))
print(predictionFig4_Up, vp = vplayout(1:3, 1))
print(predictionFig4_Down, vp = vplayout(4:5, 1))
dev.off()

# Figure 5 - Prediction line plot with std
dataIndexPredictionDynamicStd = dataIndexPrediction %>% 
  mutate(predict = c(close[1],close[1:(nrow(dataIndexPrediction)-1)]*exp(ret[2:nrow(dataIndexPrediction)])),
         predict_n1 = c(close[1],close[1:(nrow(dataIndexPrediction)-1)]*
                          exp(ret[2:nrow(dataIndexPrediction)] - std[2:nrow(dataIndexPrediction)])),
         predict_n2 = c(close[1],close[1:(nrow(dataIndexPrediction)-1)]*
                          exp(ret[2:nrow(dataIndexPrediction)] - 2*std[2:nrow(dataIndexPrediction)])),
         predict_p1 = c(close[1],close[1:(nrow(dataIndexPrediction)-1)]*
                          exp(ret[2:nrow(dataIndexPrediction)] + std[2:nrow(dataIndexPrediction)])),
         predict_p2 = c(close[1],close[1:(nrow(dataIndexPrediction)-1)]*
                          exp(ret[2:nrow(dataIndexPrediction)] + 2*std[2:nrow(dataIndexPrediction)]))) %>% 
  select(time,close,predict_n2,predict_n1,predict,predict_p1,predict_p2)
dataIndexPredictionDynamicStdPlot = melt(dataIndexPredictionDynamicStd,id.vars = 'time')

predictionFig5 = ggplot(data = dataIndexPredictionDynamicStdPlot,aes(x = as.Date(time),y = value)) + 
  geom_line(aes(color = variable,linetype = variable)) + 
  scale_color_manual('',
                     labels = c('historical','predicted - 2 std.','predicted - 1 std.',
                                'predicted','predicted + 1 std.','predicted + 2 std.'),
                     values = c('#00529e','#969696','#b51b2d','#b51b2d','#b51b2d','#969696')) + 
  scale_linetype_manual('',
                        labels = c('historical','predicted - 2 std.','predicted - 1 std.',
                                   'predicted','predicted + 1 std.','predicted + 2 std.'),
                        values = c(1,4,2,1,2,4)) + 
  theme_bw() + 
  theme(panel.background = element_blank()) + 
  theme(panel.border = element_blank()) + 
  theme(axis.line = element_line(size = 0.3)) + 
  theme(axis.ticks.x = element_blank()) + 
  theme(legend.key = element_rect(color = '#ffffff')) + 
  theme(legend.background = element_blank()) + 
  labs(x = '',y = '',title = '')
print(predictionFig5)
ggsave('predictionFig5.pdf',predictionFig5,width = 12,height = 4)
