library(ggplot2)


df1 <- read.csv('../experiments/exp-static.csv')
df2 <- read.csv('../experiments/exp-mobile.csv')

levels(df1$type) <- list(square='block', sparse='sparse')
df2$type <- 'moving-square'

df <- rbind(df1, df2)

head(df)

df <- df[df$fraction != 0.01, ]

df$fraction <- as.factor(df$fraction)
df$proportion <- as.factor(df$proportion)


ggplot(df, aes(x=proportion, y=acc_decay, color=type, shape=type)) +
    geom_point() + facet_grid(fraction ~ .) +
    xlab('Proportion of modified pixels') +
    ylab('Mean accuracy decay (3 trials)') +
    labs(color='Attack type', shape='Attack type')
ggsave('mean_acc.png')

ggplot(df, aes(x=proportion, y=patch_success_rate, color=type, shape=type)) +
    geom_point() + facet_grid(fraction ~ .) +
    xlab('Proportion of modified pixels') +
    ylab('Mean triggering rate (3 trials)') +
    labs(color='Attack type', shape='Attack type')
ggsave('mean_rate.png')