data1 <- read.csv("bunnyMiddle.csv")
data2 <- read.csv("bunnyComplex.csv")

# data are not from normal distribution*( shoking I know)
ks.test(data1$pomer_polomerov, "pnorm", mean = mean(data1$pomer_polomerov), sd = sqrt(var(data1$pomer_polomerov))) # p-value < 2.2e-16 not normal
ks.test(data2$pomer_polomerov, "pnorm", mean = mean(data2$pomer_polomerov), sd = sqrt(var(data2$pomer_polomerov))) #  p-value < 2.2e-16 not normal
hist(data1$pomer_polomerov)
hist(data2$pomer_polomerov)
qqnorm(data1$pomer_polomerov, pch = 1, frame = FALSE)# vyzera ze je posunuty
qqline(data1$pomer_polomerov, col = "steelblue", lwd = 2)
qqnorm(data2$pomer_polomerov, pch = 1, frame = FALSE)# vyzera ze je posunuty
qqline(data2$pomer_polomerov, col = "steelblue", lwd = 2)


#
ks.test(data1$pomer_polomerov, "pnorm", mean = mean(data1$pomer_polomerov), sd = sqrt(var(data1$pomer_polomerov))) # p-value < 2.2e-16 not normal

# h0 complex = middle h1 compplex != middle
wilcox.test(data1$pomer_polomerov,data2$pomer_polomerov, alternative = "two.sided")
#p-value < 2.2e-16 # h0 rejected

# h0 middle >= complex h1 midlle <= complex
wilcox.test(data1$pomer_polomerov,data2$pomer_polomerov, alternative = "less")
#  p-value = 1 h0 not rejected

wilcox.test(data1$pomer_extrem_stran,data2$pomer_extrem_stran, alternative = "two.sided")
# p-value = 0.3704 h0 not rejected

wilcox.test(data1$pomer_polomer_polobvod,data2$pomer_extrem_stran, alternative = "two.sided")
# p-value < 2.2e-16 h0 rejected
wilcox.test(data1$pomer_stran,data2$pomer_stran, alternative = "two.sided")
# p-value < 2.2e-16 h0 rejected
wilcox.test(data1$pomer_opisana_strana,data2$pomer_opisana_strana, alternative = "two.sided")
# p-value < 2.2e-16 h0 rejected

cor(data1$obsahy,data1$min_uhol)

