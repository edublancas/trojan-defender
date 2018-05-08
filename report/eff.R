library(ggplot2)


df1 <- read.csv('../experiments/exp-static.csv')
df2 <- read.csv('../experiments/exp-mobile.csv')

levels(df1$type) <- list(Square='block', Sparse='sparse')
df2$type <- 'Mobile Square'

df <- rbind(df1, df2)

head(df)

df <- df[df$fraction != 0.01, ]

df$fraction <- paste(as.character(df$fraction * 100), '% Poisoned', sep='')
df$fraction <- as.factor(df$fraction)
df$fraction <- factor(df$fraction, levels(df$fraction)[c(4, 1:3)])

df$proportion <- paste(as.character(df$proportion * 100), '%', sep='')
df$proportion <- as.factor(df$proportion)

theme <- theme(axis.text=element_text(size=12),
          axis.title=element_text(size=14),
          legend.text=element_text(size=12),
          legend.title=element_text(size=14),
          legend.position='bottom',
          strip.text=element_text(size=12))

ggplot(df, aes(x=proportion, y=acc_decay, color=type, shape=type, group=type)) +
    geom_point(size=2.5) +
    geom_line() +
    facet_wrap(~fraction, ncol=1) +
    xlab('Percentage of modified pixels') +
    ylab('Mean accuracy decay (3 trials)') +
    labs(color='Attack type', shape='Attack type') +
    scale_y_continuous(labels = scales::percent) +
    theme

ggsave('mean_acc.png', height=10, width=5)

ggplot(df, aes(x=proportion, y=patch_success_rate, color=type, shape=type, group=type)) +
    geom_point(size=2.5) +
    geom_line() +
    facet_wrap(~fraction, ncol=1) +
    xlab('Percentage of modified pixels') +
    ylab('Mean triggering rate (3 trials)') +
    labs(color='Attack type', shape='Attack type') +
    scale_y_continuous(labels = scales::percent) +
    theme

ggsave('mean_rate.png', height=10, width=5)