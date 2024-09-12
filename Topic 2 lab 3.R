# Topic 2 (Lab 3)



rm(list=ls())
gc()
cat("\f")

# Libraries
# =========
library(wooldridge) # Wooldridge.
library(moments) # moments of a probability distribution.
library(nortest) # normality testing.
library(tidyverse) # data wrangling.
library(lmtest) # diagnostic checking in linear regression models.
library(sandwich) # model-robust covariance matrix estimators.
search() # active packages in my machine.

data(wage2)
ls()
? wage2
attach(wage2)

# Question (1): level-level model
#=============
lev_lev <- lm(wage ~ educ + exper + I(exper^2) + tenure)
summary(lev_lev)

# Question 1.1: Residuals normality
#=============
plot(lev_lev)
dev.off()
a = resid(lev_lev) # store residuals in object a.
summary(a, digits=4)
skewness(a) # positive skew.
boxplot(a) # many outliers.
kurtosis(a) # Leptokurtic.
shapiro.test(a) # H0 rejected.
ad.test(a) # H0 rejected.
lillie.test(a) # H0 rejected.
dev.off()

# Question 1.2: Gauss-Markov
# ============
# MLR 1: linearity - satisfied (see residuals plots).
# MLR 2: random sample - may be NOT (students: why?)
# MLR 3: no perfect collinearity (see below) - satisfied.
# MLR 4: zero conditional mean - may be NOT.
# We cannot measure innate ability yet we are estimating a wage model.
# MLR 5: homoskedasticity - may be NOT (see below).

# MLR 3 - No perfect collinearity in regressors
# ==============================================
summary(lm(educ ~ exper))
cor(educ,exper)
summary(lm(educ ~ tenure))
cor(educ,tenure)
summary(lm(exper ~ tenure))
cor(exper,tenure)

# MLR 5  - Homoskedasticity
# ==========================
plot(wage ~ educ) # plot funnels-out.
plot(wage ~ exper)# plot funnels-out.
dev.off()

# Question 1.3: make a point estimate prediction for wage.
#=============
# point estimate: 10 years education, 8 years experience, 2 years tenure.
# ==============
# Q1.3.1: Manual prediction.
# ======
pred_val = sum(lev_lev$coef*c(1,10,8,64,2))

# Q1.3.2: Use the 'shift the starting point' trick (pages 207-208).
# ======
wage_aux <-lm(wage ~ I(educ-10) + I(exper-8) + I((exper-8)^2) + I(tenure-2))
summary(wage_aux)
wage_est_trick = summary(wage_aux)$coef[1,1]
# confirm equality of the two predictions:
rbind("manual"=pred_val, "trick"=wage_est_trick)

# Question (1.4): construct 95% CIs around the point prediction.
# ==============
# Critical values of the t-distribution
# ======================================
qt(.025, wage_aux$df, lower.tail = TRUE) # 5% two-sided lower.
qt(1-.025, wage_aux$df, lower.tail = TRUE) # 5% two-sided upper,
qt(.975, wage_aux$df, lower.tail = TRUE) # 5% two-sided upper.

# Q1.4.1: Manual 95% CI (remove 2.5% from either tail).
# ======
beta=summary(wage_aux)$coef[1,1] # extract b0 (intercept).
sd=summary(wage_aux)$coef[1,2] # extract SE of b0 (intercept).
low=beta-1.96*sd  # critical value of the t (2.5th percentile) = -1.96.
high=beta+1.96*sd # critical value of the t (97.5th percentile) = +1.96.
cbind(low,high)

# Q1.4.2: Use R-inbuilt 95% CIs.
# ======
confint(wage_aux,level=0.95) # 95% CI's for all coefficients.
a = confint(wage_aux,level=0.95) ["(Intercept)",] # 95% CI's only for the intercept.

# Question 1.5: construct  95% CIs around mean values of X (i.e., find the tightest CIs).
# ============
wage_aux_mean <-lm(wage~I(educ-mean(educ)) + I(exper-mean(exper)) + I((exper-mean(exper))^2)
                   + I(tenure-mean(tenure)))
summary(wage_aux_mean)
confint(wage_aux_mean,level=0.95)
# Demonstrate: the tightest CI are at the mean values.
# ===========
b = confint(wage_aux_mean,level=0.95)["(Intercept)",] # 95% CI's for the intercept.
rbind("CI_@obs"=a,"CI_mean"=b) # tightest CI's at mean values.

# Question 2: log-level model estimation.
# ===========
log_lev <- lm(log(wage) ~ educ + exper + I(exper^2) + tenure)
summary(log_lev)$coef

# Question 2.1: wage and log(wage) histograms to deduce advantages of using logs.
# ============
hist(wage, breaks=50,freq=FALSE, col="red")
plot(function(x) dnorm(x, mean = mean(wage), sd = sd(wage), log = FALSE), min(wage),max(wage),
     main = "hist wage and normal curve, level",add=T)
# print a normal curve over the data with same mean and SD as the data in levels.
hist(log(wage),breaks=50,freq=FALSE, col="green")
plot(function(x) dnorm(x, mean = mean(log(wage)), sd = sd(log(wage)), log = FALSE),
     min(log(wage)),max(log(wage)), main = "hist log(wage) and normal curve logs",add=T)
dev.off()
par(mfrow=c(1,2))
hist(wage, breaks=50,freq=FALSE, col="red")
plot(function(x) dnorm(x, mean = mean(wage), sd = sd(wage), log = FALSE), min(wage),max(wage),
     main = "hist wage and normal curve, level",add=T)
hist(log(wage),breaks=50,freq=FALSE, col="green")
plot(function(x) dnorm(x, mean = mean(log(wage)), sd = sd(log(wage)), log = FALSE),
     min(log(wage)),max(log(wage)), main = "hist log(wage) and normal curve logs",add=T)
dev.off()

# The log transformation compresses.
# log(y) gives a better distribution of the error [i.e., observed(y)-predicted(y)] term.
# A better distribution - more reliable t-stats and hypothesis tests.

# Question 2.2: predicting y from log(y)[pages 212-213].
#=============
# Steps
# =====
# (i) Predict log(y).
#             =====
# (ii) Estimate adjustment parameters (3 variants).
#               ==========
# (iii) Use (i) and (ii) to predict y from log(y).
#                                   =
# (2.2.1) Predict log(y).
#         ==============
# Manual prediction [equation 6.39]:
# =================
pred_logy = sum(log_lev$coef*c(1,10,8,64,2))
# "Shift the starting point" prediction
# ====================================
log_lev_pred <- lm(log(wage) ~ I(educ-10) + I(exper-8) + I((exper-8)^2) + I(tenure-2))
Elogy = coef(log_lev_pred)[1] # extract intercept of "log_lev_pred".
# verify "pred_logy" = "Elogy"
# ===========================
rbind("manual" = pred_logy, "shift" = Elogy)
# Exponentiate log(y) to obtain unadjusted y prediction.
# ============
y_unadj = exp(Elogy) # intercept from "shift the starting point" prediction.

# (2.2.2) Estimate adjustment parameter.
#         =============================
# Normality based adjustment
# ==========================
summary(log_lev$res) # "log_lev" residuals.
var(log_lev$res) # "log_lev" residuals variance.
alpha_normal = exp(var(log_lev$res)/2) # [equation 6.40].
# Smearing based adjustment
# =========================
alpha_smear = sum(exp(log_lev$res))/length(wage) # [equation 6.43].
# Regression based adjustment [page 213: equation 6.44].
# ==========================
mi <- exp(predict(log_lev)) # exponentiate predicted log(y).
aux <- lm(wage ~ mi-1) # regress wage on mi in a regression without intercept.
alpha_reg = summary(aux)$coef[1,1] # extract coefficient from object "aux".
# (2.2.3) Use adjustments to predict y from log(y).
#         =========================================
predict_alpha_normal=alpha_normal*y_unadj
predict_alpha_smear=alpha_smear*y_unadj
predict_alpha_reg=alpha_reg*y_unadj
# 2.2.4 Confirm unadjusted prediction (y_unadj) underestimates y.
#       ========================================================
rbind(y_unadj, predict_alpha_normal,predict_alpha_smear, predict_alpha_reg)

# Question 2.3: 95% CI around point prediction.
#=============
# Regression "log_lev_pred" uses "shift the starting point" trick.
# ===============================
summary(log_lev_pred)$coef
# 2.3.1 Un-adjusted CI
# ==============
s=confint(log_lev_pred) # 95% CI for "log_lev_pred".
low = s[1,1] # extract lower limit - intercept only.
high = s[1,2] # extract upper limit - intercept only.
cbind(low,high) # 95% CI for predicted log(y).
# 2.3.2 Adjusted CI
# ===========
# Adjust 95% CI boundaries to avoid bias.
# Use alpha normal to demonstrate  procedure.
# Adjusted CI manually:
# ====================
low_alpha_normal = exp(low)*alpha_normal # adjust the lower CI - intercept only.
high_alpha_normal = exp(high)*alpha_normal # adjust the upper CI - intercept only.
rbind(low_alpha_normal, high_alpha_normal)
# Other adjustments
# =================
s = confint(log_lev_pred) # store 95% CI for all coeffs of "log_lev_pred" in s.
cnfint_normal = (exp(s)*alpha_normal)[1,] # alpha normal on intercept.
cnfint_smear = (exp(s)*alpha_smear)[1,] # alpha smear on intercept.
cnfint_reg = (exp(s)*alpha_reg)[1,] # alpha reg on intercept.
rbind("normal"=cnfint_normal,"smear"=cnfint_smear,"reg"=cnfint_reg)

# Question 2.4: Selecting between the level_level and log_level models.
#===========
# 2.4.1 Alternative R-squared (log-lev model)
# ===========================================
# From the log_level model, we can use code from Q2.2 to predict y from log y.
mi <- exp(predict(log_lev)) # exponentiate predicted log(y).
aux <- lm(wage ~ mi-1) # regression without intercept.
alpha_reg = summary(aux)$coef[1,1] # extract "alpha_reg".
predict_logy = predict(log_lev) # predict log(wage).
predict_y    = alpha_reg*exp(predict_logy) # predict y from log(y)
# We can now compute alternative R-square for the log-level model:
# ====================
alt_r_sq_log = cor(predict_y,wage)^2
# 2.4.2 R-square (lev_lev model)
# ========================
# (i) "Canned" R2
# ===========
a = summary(lev_lev)$r.squared # retrieve R-squared from the level-level estimation.
# (ii) Alternative R-square
# ====================
y1hat = predict(lev_lev) # predict yhat.
b = cor(wage,y1hat)^2 # alternative R-square for lev_lev.
# (iii) Confirm (i) = (ii).
# ==================
c = b - a # the difference (e-17) is virtually zero!
# The two approaches produce exactly the same results.
# 2.4.3 lev_lev vs log_level
# ====================
r_sq_level = summary(lev_lev)$r.squared
rbind("level_level"=r_sq_level,"log_level" = alt_r_sq_log)
# Remember: we want to select between level-level and log-level.
#=========
# Level-level makes slightly better predictions (select the level-level model).
# Question 2.5: based on the t-value.
#============
# Two sided t-test on tenure at the 5% level.
# Calculate t:
# ===========
a = summary(log_lev)$coef["tenure",1] # store tenure beta in a.
b = summary(log_lev)$coef["tenure",2] # store tenure SE in b.
t = a/b # = 5.164544
# Degrees of freedom:
# ==================
summary(log_lev)$df # extract df = n-k-1.
summary(log_lev)$df[2]
# Critical t-value:
# ================
qt(.025,930) # two-sided lower 5%.
qt(.975,930) # two-sided upper 5%.
# Test - does 5.164544 fall in the rejection region?
# ===
# t= 5.164544 > critical value for the 5% level.
# t= 5.164544 falls in the H0 rejection region.
# Tenure is significant at the 5% level (significant even at the 1% level).
qt(.005,930) # two-sided lower 1% (= (1/100)/2).
qt(.995,930) # two-sided upper 1%.
# Areas under the t-distribution:
# ===============================
# p-value: under H0, what is the prob. of t = 5.164544 or a more extreme value?
# =======
# one tailed (t = 5.164544 with log_lev$df (= 930))
# ==========
pt(t,log_lev$df, lower.tail=T) # area to the left.
1-pt(t,log_lev$df, lower.tail=T) # area to the right.
# two tailed
# ==========
2*(1-pt(t,log_lev$df, lower.tail=T))
# Question 2.6: the p-value
# ============
2*(1-pt(t,log_lev$df, lower.tail=T))
# Conclusion:
# ==========
# p-value = 0.00000029486 (significant even at the 0.00003% level).
# Assuming H0 is true, prob.(getting t = 5.164544 or a more extreme value) = 0.00003%.
# It can happen 3 times in 100,000 trials.
# 0.00000029486 < 0.05 (falls in the rejection region).

# Question 2.7: based on the 95% CI
#=============
# Critical values:
qt(.05/2,log_lev$df) # 5% two-sided lower.
qt(1-.05/2,log_lev$df) # 5% two-sided upper.
# Manual CIs:
# ======
beta=summary(log_lev)$coef["tenure",1]
sd=summary(log_lev)$coef["tenure",2]
low=beta-1.96*sd
high=beta+1.96*sd
cbind(low,high)
# Direct 95% CIs:
# =======
confint(log_lev,level=0.95) ["tenure",]
# Conclusion:
# ==========
# Zero is NOT in the 95% CI.
# Effect of tenure on wages is significantly different from zero at the 5% level.

# Question 2.8
#=============
# Question 2.8.1: two-sided alternative
# ==============
# Calculate t-stat
# ================
t1 = ((summary(log_lev)$coef["educ",1])-0.05)/summary(log_lev)$coef["educ",2]
# Hypothesis H0: beta=0.05 vs. H1: beta not= 0.05.
# =============
# 95% confidence: remove 2.5% from either tail.
2*(1-pt(t1,log_lev$df, lower.tail=T))
# Assuming H0 is true, prob. of getting a more extreme value than t= 3.759523 is 0.0001808616.
# Since 0.0001808616 < 0.05, p falls in the H0 rejection region.
# Conclude:beta is NOT equal to 0.05.
# ====================
# Question 2.8.2 H0: beta > 0.05 vs. H1: beta < 0.05 (left-sided alternative).
# =================
# If H0: beta > 0.05, I can only reject H0 on the left tail.
# t-distribution: remove 5% from the left tail.
# ==========
pt(t1,log_lev$df,lower.tail=T)
# Since 0.9999096 > 0.05, p falls outside the H0 rejection region.
# Provides evidence in SUPPORT of H0: beta > 0.05.
# Conclude: beta > 0.05.
# ============
# # Question 2.8.3 H0: beta < 0.05 vs. H1: beta > 0.05 (right-sided alternative)
# ====================
# t-distribution: remove 5% from the right tail.
1-(pt(t1, log_lev$df,lower.tail=T))
# Under H0, the prob. of getting a more extreme value than t= 3.759523 is 9.043078e-05.
# Conclude:  9.043078e-05 < 0.05 (reject H0: beta < 0.05).

# Question 2.9:
#=============
summary(log_lev) # log_lev estimation results.
# Is the marginal impact of tenure equal to that of education?
# Theory might suggest this is true - to be sure, we test.
# Auxilliary equation to estimate:
wage_aux <- lm(log(wage) ~ educ + exper + I(exper^2) + I(tenure + educ))
summary(wage_aux)$coef
# t-test on education:
educ_stats = summary(wage_aux)$coef["educ", ]
# Education is statistically significant.
# Marginal impact of tenure on wages differs from that of education.

# Question 3: Elasticities
# ==========
# Estimate a log-log1 model i.e., add log(hours) as covariate.
log_log1 <-lm(log(wage) ~ educ + exper + I(exper^2) + tenure + log(hours))
summary(log_log1)
summary(log_log1)$coef["log(hours)",]
# 1% increase in hours worked decreases wages by 0.17% (or increases wage by -0.17%)
# ===========                                    =====

# Question 4. restricted least squares
# =========
# Add dummies for residence areas "south" and "urban" to model log-log1.
log_log2 <- lm(log(wage) ~ educ + exper + I(exper^2) + tenure + log(hours) + south + urban)
summary(log_log2)
summary(log_log2)$coef["south", ]
summary(log_log2)$coef["urban", ]
# South and urban are individually very statistically significant.

# Question 4.1: restricted least squares
#=============
# H0: beta6("south")= 0  and beta7 ("urban") = 0 (joint significance).
anova(log_log1,log_log2)
# Form of F-test on the slides.
# The unrestricted model significantly improves on the restricted model.
# H0 (south and urban add zero to model "log_log1") is strongly rejected.

# Question 5: Heteroskedasticity (HS)
#===========
# Question 5.1: residuals plot.
#===========
plot(log_log2, lwd=3)
stdres = log_log2$res # store residuals in object "stdres".
fit = log_log2$fitted # store fitted values in object "fit".
plot(fit,stdres) # (no obvious HS signs).
plot(fit,abs(stdres)) # (more obvious HS signs).
par(mfrow=c(1,2))
plot(fit,stdres)
plot(fit,abs(stdres))
dev.off()
# Plots are only indicative.
# Need for formal tests.

# Formal tests
# ============
skewness(stdres)
kurtosis(stdres)
shapiro.test(stdres)
ad.test(stdres)
lillie.test(stdres)

# Question 5.2: Breusch-Pagan test
#=============
# Null hypothesis
# ===============
# H0: homoskedasticity (zero corr between residuals variance and X).
# The F distribution:
# ===================
# qf() computes the quantile (critical value) for a given area.
# k = # of predictors.
# n = # of obs.
# Numerator DF (df1) = k.
# Denominator DF (df2) = (n-k-1).
# Example: alpha= 0.05, df1 = 6, and df2 = 8.
# =======
qf(p=.05, df1=6, df2=8, lower.tail=FALSE)
# 5.2.1 Manual BP test
# ====================
res = log_log2$res # store residuals of log_log2 in res.
res2 = res^2 # store residuals variance in res2.
aux1 <- lm(res2 ~ educ + exper + I(exper^2) + tenure + log(hours) + south + urban)
summary(aux1)
# Evidence for HS - 3 variables are individually statistically significant.
# (tenure, log(hours) and south).
# Degrees of freedom:
length(wage) # 935 obs.
df1 = 7 # 7 predictors
df2 = 927 # 935-7-1
# F calculated:
F_calculated = (summary(aux1)$r.squared/df1)/((1-summary(aux1)$r.squared)/df2)
# Critical F value:
qf(p=.05, df1, df2, lower.tail=FALSE) # F_critical = 2.01944.
# F_calculated > F_critical (lies in the H0 rejection region).
# p-value:
# =======
pf(5.767, df1, df2, lower.tail = FALSE) # (p_calculated = 1.487575e-06).
# p_calculated < 0.05 (lies in the H0 rejection region).
# Conclude: reject H0 of homoskedasticity.
# ========
# 5.2.2 R-canned test
# ===================
bptest(log_log2)
# strongly rejects H0 of homoskedasticity.

# Question 5.3: White test
#=============
fit = log_log2$fit # save fitted values of log_log2.
aux2 <- lm(res2 ~ fit + I(fit^2)) # estimate auxiliary regression to the 2nd power.
summary(aux2)
# F-test
# ======
df1=2
df2=932
F_calculated = (summary(aux2)$r.squared/df1)/((1-summary(aux2)$r.squared)/df2)
qf(p=.05, df1, df2, lower.tail=FALSE)
pf(3.133435, df1, df2, lower.tail = FALSE)
# Conclude
# ========
# The White test rejects homoskedasticity at the 5% level.
# Conclusions of the White are not as strong as those reached by Breusch-Pagan.
# Both tests reject H0 - there is evidence for heteroskedasticity.
# Possible solution:
# =================
# 1. Change functional forms for problem variables?
# 2. Use HS robust SEs.

# Question 5.4: Regular vs. HS robust SEs
#=============
# Regular SEs
# ===========
# First, retrieve the variance-covariance matrix.
vcov(log_log2)
# Retrieve diagonal of the variance-covariance matrix.
diag(vcov(log_log2))
# Finally, square root the diagonal.
std.err <- sqrt(diag(vcov(log_log2))) # regular SEs.
# Confirm equality of the computed and canned SEs:
cbind("computed SEs" = std.err, "canned SEs" = summary(log_log2)$coef[, 2])
# Robust SEs
# ==========
robust.std.err<-sqrt(diag(vcovHC(log_log2, type = "HC")))
# regular vs. HS robust SEs
# =========================
cbind("regular"=std.err,"robust"=robust.std.err)
# Under HS - regular SEs are more precise (smaller) than they ought to be.
# log(hours) has biggest change moving from regular to robust SEs.
# Question - what is the problem with using regular SEs under HS?
# ========
# Under HS, OLS remains unbiased and consistent.
# But the likelihood of committing error type I becomes higher.
# More likely to reject H0 when it is true.
# To make this point, isolate the log(hours) stats.
# ===================
a_regular_SE = coeftest(log_log2)[6,]
b_robust_SE = coeftest(log_log2,vcov = vcovHC(log_log2, type = "HC"))[6,]
rbind("regular" = a_regular_SE,"robust" = b_robust_SE)
# regular SEs - log(hours) is significant at the 5%.
# robust SEs - log(hours) is only significant at the 10%.
# Note some alternative commands for SEs:
# =======================================
coeftest(log_log2) # regular SEs.
coeftest(log_log2,vcov = vcovHC(log_log2, type = "HC")) # robust SEs.
# Putting everything together:
# ===========================
options(scipen=2)
a = coeftest(log_log2)[,1]
b = coeftest(log_log2,vcov = vcovHC(log_log2, type = "HC"))[,1]
c = coeftest(log_log2)[,2]
d = coeftest(log_log2,vcov = vcovHC(log_log2, type = "HC"))[,2]
e = c/d
cbind("beta_reg" = a,
      "beta_rob" = b,
      "se_reg"   = c,
      "se_rob"   = d,
      "(se_reg)/(se_rob)" = e)
# HS has NO impact on bias.
# HS has an impact on precision.

# Question 5.5: Should we worry about HS going forward?
#=============
# Breusch-Pagan suggests a strong HS problem.
# White suggests a weak HS problem.
# Should we worry about HS?
summary(aux1) # BP test output.
# Problem variables - significant correlate with residuals variance.
# =================
summary(aux1)$coef["tenure",]
summary(aux1)$coef["log(hours)",]
summary(aux1)$coef["south",]
# Explore these variables further using residual plots:
dev.off()
plot(tenure, abs(log_log2$res))
# Error variance decreases with tenure (supports negative slope).
# Tenure exhibits strong HS.
dev.off()
plot(log(hours), abs(log_log2$res))
# HS not as clear.
# But there seems to be a positive correlation.
dev.off()
plot(south, abs(log_log2$res))
# Since south is a dummy, the plot is not informative.
# We can however use the means instead.
# south: =1 if live in south
south_0_mean = mean(abs(log_log2$res)[south==0])
south_1_mean = mean(abs(log_log2$res)[south==1])
rbind("south"=south_1_mean,"otherwise"= south_0_mean)
# The mean or average variance is very different.
# Variance is much higher for those living in the south.

# Evidence in favor of HS.
# Conclude: we might be better off with HS robust SEs.
# ========

# Question 5.6: repeat the test in Q4 with robust SEs.
#=============
# In Q4, we used restricted LS to jointly test H0: beta6 = 0 and beta7 = 0.
anova(log_log1,log_log2) # we rejected H0.
# Alternative code:
# =================
# ?waldtest()
waldtest(log_log1, log_log2)
waldtest(log_log1, log_log2, vcov=vcov(log_log2)) # regular SEs.
# H0: south and urban add zero to explaining variance.
# Strong rejection of H0.
# Do we get the same results with robust SEs?
# ==========
waldtest(log_log1, log_log2, vcov=vcovHC(log_log2, method="HC"))
# Rejection of H0 is ALMOST as strong.
# Conclude: we should use robust SEs.
# =========

# Prelude to Q6: Omitted Variables Bias (OVB) or Endogeneity
# =============
# A problem we will address in panel data (+ instrumental variables).
# ==========    =====================
# Recall from Question 4 where we EXCLUDED IQ score from the estimation:
log_log2 <- lm(log(wage) ~ educ + exper + I(exper^2) + tenure + log(hours) + south + urban)
summary(log_log2)$coef
# In Question 6, we add IQ score as a proxy for ability.
# log-log3 model <- add "IQ score" to log_log2.
# Question: what is the impact of excluding IQ score on BIAS?
# ========

# Question (6):
# ============
log_log3 <-lm(log(wage) ~ educ + exper + I(exper^2) + tenure + log(hours) + south + urban + IQ)
summary (log_log3)$coef
# Explain cbind() + where I include NA.
options(scipen=2)
cbind('without IQ'   =c(log_log2$coef,NA),
      'with IQ'      =log_log3$coef,
      '% change'     =c( 100*(log_log3$coef[-9]-log_log2$coef)/log_log2$coef, NA),
      't stat -IQ'   =c(summary(log_log2)$coef[,3], NA),
      't stat +IQ'   =summary(log_log3)$coef[,3])
# The % change in the effect of experience^2 is very large.
# Note what happens to the t-stat of experience^2.
# Among the significant variables, education drops by 23% and south by 20%.
# Might indicate a corr between this further:
summary(lm(IQ ~ educ + south))$coef
# In this data set, people with higher education also have higher IQ (on average).
# People living in the south have lower IQ (on average).
# This means that in the estimation without IQ, the positive effect of education was overestimated.
# =============                                          =============
# The negative effect of living in the south was also overestimated.
# =============

# Ending your R session
# =====================
detach(wage2)
dev.off()
rm(list=ls())
gc()
cat("\f")
