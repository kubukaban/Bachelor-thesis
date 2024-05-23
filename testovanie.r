data1 <- read.csv("csv/bunnyMiddle.csv")
data2 <- read.csv("csv/bunnyComplex.csv")

# data are not from normal distribution*( schoking I know)
#rho
ks.test(data1$Radius.ratio, "pnorm", mean = mean(data1$Radius.ratio), sd = sqrt(var(data1$Radius.ratio))) # p-value < 2.2e-16 not normal
ks.test(data2$Radius.ratio, "pnorm", mean = mean(data2$Radius.ratio), sd = sqrt(var(data2$Radius.ratio))) #  p-value < 2.2e-16 not normal
#tau
ks.test(data1$Edge.ratio, "pnorm", mean = mean(data1$Edge.ratio), sd = sqrt(var(data1$Edge.ratio))) # p-value < 2.2e-16 not normal
ks.test(data2$Edge.ratio, "pnorm", mean = mean(data2$Edge.ratio), sd = sqrt(var(data2$Edge.ratio))) #  p-value < 2.2e-16 not normal
#nu
ks.test(data1$Circumradius.to.half.perimeter, "pnorm", mean = mean(data1$Circumradius.to.half.perimeter), sd = sqrt(var(data1$Circumradius.to.half.perimeter))) # p-value < 2.2e-16 not normal
ks.test(data2$Circumradius.to.half.perimeter, "pnorm", mean = mean(data2$Circumradius.to.half.perimeter), sd = sqrt(var(data2$Circumradius.to.half.perimeter))) #  p-value < 2.2e-16 not normal
#iota
ks.test(data1$Aspect.ratio, "pnorm", mean = mean(data1$Aspect.ratio), sd = sqrt(var(data1$Aspect.ratio))) # p-value < 2.2e-16 not normal
ks.test(data2$Aspect.ratio, "pnorm", mean = mean(data2$Aspect.ratio), sd = sqrt(var(data2$Aspect.ratio))) #  p-value < 2.2e-16 not normal
#omega
ks.test(data1$Circumradius.to.edge, "pnorm", mean = mean(data1$Circumradius.to.edge), sd = sqrt(var(data1$Circumradius.to.edge))) # p-value < 2.2e-16 not normal
ks.test(data2$Circumradius.to.edge, "pnorm", mean = mean(data2$Circumradius.to.edge), sd = sqrt(var(data2$Circumradius.to.edge))) #  p-value < 2.2e-16 not normal

hist(data1$Radius.ratio)
hist(data2$Radius.ratio)


library(ggplot2)
ggplot(data1, aes(x = Radius.ratio)) +
  geom_histogram(binwidth = 0.1, fill = "blue", color = "black") +
  scale_y_log10() + 
  scale_x_log10()+
  ggtitle("Histogram of Radius Ratiofor BunnyMiddle with Logarithmic Y-Axis") +
  xlab("Radius Ratio") +
  ylab("Frequency")


ggplot(data2, aes(x = Radius.ratio)) +
  geom_histogram(binwidth = 0.1, fill = "blue", color = "black") +
  scale_y_log10() +
  scale_x_log10() +
  ggtitle("Histogram of Radius Ratio for BunnyComplex with Logarithmic Y-Axis") +
  xlab("Radius Ratio") +
  ylab("Frequency")

# h0 complex = middle h1 compplex != middle
wilcox.test(data1$Radius.ratio,data2$Radius.ratio, alternative = "two.sided")
#p-value < 2.2e-16 # h0 rejected

# h0 middle >= complex h1 midlle <= complex
wilcox.test(data1$Radius.ratio,data2$Radius.ratio, alternative = "less")
#  p-value = 1 h0 not rejected

wilcox.test(data1$Edge.ratio,data2$Edge.ratio, alternative = "two.sided")
# p-value = 0.3704 h0 not rejected

wilcox.test(data1$Circumradius.to.half.perimeter,data2$Circumradius.to.half.perimeter, alternative = "two.sided")
# p-value < 2.2e-16 h0 rejected

wilcox.test(data1$Aspect.ratio,data2$Aspect.ratio, alternative = "two.sided")
# p-value < 2.2e-16 h0 rejected
wilcox.test(data1$Circumradius.to.edge,data2$Circumradius.to.edge, alternative = "two.sided")
# p-value < 2.2e-16 h0 rejected

