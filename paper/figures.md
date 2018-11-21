# Building figures from paper

```r
library(ggplot2)
library(grid)
library(gridExtra)
library(cowplot) # for combining figures
library(plyr)
require(dplyr)
require(scales)

themepurple = "#d0c1ff"
themeorange = "#ffab03"
themeyellow = "#fff7c1"
themeoffwhite = "#f2f6ff"
themedarkgrey = "#565656"

themedarktext = "#707070"
yearline = "black"
yearline_size = 0.5
big_fontsize = 12

integrated_legend = theme(
  legend.position = c(0.35, 0.7),
  legend.background = element_rect(fill=themeoffwhite, size=0.5, linetype="solid"),
  legend.text = element_text(size=big_fontsize)
)
better_line_legend = guides(color = guide_legend(override.aes = list(size = 4)))

setwd('/Users/rabdill/code/rxivist/paper/data') # *tk remove this
```






## Figure 1

### Figure 1a: Median downloads per category
```sql
SELECT d.article, d.downloads, REPLACE(a.collection, '-', ' ') AS collection
FROM paper.alltime_ranks d
INNER JOIN paper.articles a ON d.article=a.id
WHERE a.collection IS NOT NULL;
```

```r
paperframe = read.csv('downloads_per_category.csv')

medianplot <- ggplot(data=paperframe, aes(
  x=reorder(collection, downloads, FUN=median),
  y=downloads,
  fill=collection)) +
geom_boxplot(outlier.shape = NA, coef=0) +
scale_y_continuous(labels=comma) +
coord_flip(ylim=c(0,1000)) +
# coord_cartesian(ylim=c(0,1000)) +
theme_bw() +
labs(x="", y="downloads per paper") +
theme(
  panel.grid.major.y = element_blank(),
  legend.position="none",
  axis.text = element_text(size=big_fontsize, hjust=0.9)
) +
geom_hline(yintercept=median(paperframe$downloads), col=themedarkgrey, linetype="dashed", size=1)

# comparing neuroscience papers to non-neuroscience:
paperframe$isneuro <- ifelse(paperframe$collection =='neuroscience',TRUE, FALSE)
ddply(paperframe, .(isneuro), summarise, med = median(downloads))
wilcox.test(downloads~isneuro, data=paperframe)

# comparing all collections:
leveneTest(downloads~collection, data=paperframe)
kruskal.test(downloads~collection, data=paperframe)
oneway.test(downloads~collection, data=paperframe) # Welch's ANOVA
```

### Figure 1b: Distribution of downloads per paper

```sql
SELECT *
FROM (
	SELECT a.id, COALESCE(SUM(t.pdf), 0) AS downloads
	FROM paper.articles a
	LEFT JOIN paper.article_traffic t ON a.id=t.article
	WHERE a.collection IS NOT NULL
	GROUP BY a.id
	ORDER BY downloads DESC
) AS totals
WHERE downloads > 0
```

```r
distroframe = read.csv('downloads_per_paper.csv')

other <- ggplot(distroframe, aes(x=downloads)) +
geom_histogram(
  fill=themeorange,
  bins = 50
) +
scale_x_log10(labels = comma) +
scale_y_continuous(label=function(x) x/1000) +
labs(y = "papers (thousands)", x = "total downloads (log scale)") +
geom_vline(
  xintercept=median(distroframe$downloads),
  col=themedarkgrey, linetype="dashed", size=1
) +
annotate("text", x=median(distroframe$downloads)+6500, y=3250, label=paste("median:", round(median(distroframe$downloads), 2))) +
theme_bw() +
theme(
  plot.background = element_rect(fill = themeoffwhite)
)
```

### Figure 1c: Cumulative downloads over time, per category

```sql
SELECT article_traffic.year||'-'||lpad(article_traffic.month::text, 2, '0') AS date,
	SUM(article_traffic.pdf),
	REPLACE(articles.collection, '-', ' ') AS collection
FROM paper.article_traffic
LEFT JOIN paper.articles ON article_traffic.article=articles.id
GROUP BY 1,3
ORDER BY 1,3;
```
(Data organized in `downloads_per_month_cumulative.xlsx`, then moved to `downloads_per_month_cumulative.csv`)

```r
monthframe=read.csv('downloads_per_month_cumulative.csv')

# Cumulative downloads:
yearlabel = -210000

x <- ggplot(monthframe, aes(x=date, y=cumulative, group=collection, color=collection)) +
geom_line(size=1) +
labs(x = "month", y = "total downloads") +
theme_bw() +
scale_y_continuous(breaks=seq(0, 3000000, 500000), labels=comma) +
theme(
  plot.margin = unit(c(1,8,1,1), "lines"), # for right-margin labels
  axis.text.x=element_blank(),
  axis.text.y = element_text(size=big_fontsize, color = themedarktext),
  axis.title.y = element_text(size=big_fontsize),
  legend.position="none"
) +
annotation_custom(
  grob = textGrob(label = "neuroscience", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 3000000, ymax = 3050000, xmin = 61, xmax = 61) +
annotation_custom(
  grob = textGrob(label = "bioinformatics", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 2900000, ymax = 2926524, xmin = 61, xmax = 61) +
annotation_custom(
  grob = textGrob(label = "genomics", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 2714730, ymax = 2714730, xmin = 61, xmax = 61) +
annotation_custom(
  grob = textGrob(label = "genetics", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 1507955, ymax = 1507955, xmin = 61, xmax = 61) +
annotation_custom(
  grob = textGrob(label = "evolutionary biology", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 1348577, ymax = 1348577, xmin = 61, xmax = 61) +
annotation_custom(
  grob = textGrob(label = "microbiology", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 851647, ymax = 851647, xmin = 61, xmax = 61) +
annotation_custom(
  grob = textGrob(label = "cancer biology", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 659128, ymax = 659128, xmin = 61, xmax = 61) +
annotation_custom(
  grob = textGrob(label = "2014", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 3, xmax = 3) +
geom_vline(xintercept=3, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2015", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 15, xmax = 15) +
geom_vline(xintercept=15, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2016", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 27, xmax = 27) +
geom_vline(xintercept=27, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2017", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 39, xmax = 39) +
geom_vline(xintercept=39, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2018", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 51, xmax = 51) +
geom_vline(xintercept=51, col=yearline, size=yearline_size)

main <- ggplot_gtable(ggplot_build(x))
main$layout$clip[main$layout$name == "panel"] <- "off"
```

### Figure 1 combined

```r
plot_grid(medianplot, main, labels=c("(a)", "(c)"),
  label_x = .2, label_y = 0,
  hjust = -0.5, vjust = -0.5,
  ncol = 2, nrow = 1,
  align = "h"
) +
draw_plot_label("(b)", size = 14, x = 0.6, y=0.6)

print(other, vp = viewport(
  width = 0.22, height = 0.3,
  x = 0.71,
  y = 0.75
))
```





## Figure 2

```sql
SELECT EXTRACT(YEAR FROM posted)||'-'||lpad(EXTRACT(MONTH FROM posted)::text, 2, '0') AS date,
	REPLACE(collection, '-', ' ') AS collection,
	COUNT(id) AS submissions
FROM paper.articles
GROUP BY 1,2
ORDER BY 1,2;
```
(Data organized in `submissions_per_month.xlsx`, then final numbers transferred to `submissions_per_month.csv`)

### Figure 2a: Cumulative monthly submissions per category

```r
# Cumulative submissions over time
monthframe=read.csv('submissions_per_month.csv')

yearlabel1 = -380
x <- ggplot(monthframe, aes(x=date, y=cumulative, group=collection, color=collection)) +
geom_line(size=1) +
labs(x = "", y = "cumulative preprints") +
theme_bw() +
scale_y_continuous(breaks=seq(0, 6100, 1000), labels=comma) +
theme(
  plot.margin = unit(c(1,8,1,1), "lines"), # for right-margin labels
  axis.text.x=element_blank(),
  axis.text.y = element_text(size=big_fontsize, color = themedarktext),
  axis.title.y = element_text(size=big_fontsize),
  legend.position="none"
) +
annotation_custom(
  grob = textGrob(label = "neuroscience", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 6081, ymax = 6081, xmin = 61, xmax = 61) +
annotation_custom(
  grob = textGrob(label = "bioinformatics", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 4064, ymax = 4064, xmin = 61, xmax = 61
) +
annotation_custom(
  grob = textGrob(label = "evolutionary bio.", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 3200, ymax = 3200, xmin = 61, xmax = 61
) +
annotation_custom(
  grob = textGrob(label = "genomics", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 2900, ymax = 2900, xmin = 61, xmax = 61
) +
annotation_custom(
  grob = textGrob(label = "microbiology", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 2600, ymax = 2600, xmin = 61, xmax = 61
) +
annotation_custom(
  grob = textGrob(label = "genetics", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 2245, ymax = 2245, xmin = 61, xmax = 61
) +
annotation_custom(
  grob = textGrob(label = "cell biology", hjust = 0, gp = gpar(fontsize = big_fontsize)),
  ymin = 1605, ymax = 1605, xmin = 61, xmax = 61
) +
geom_vline(xintercept=3, col=yearline, size=yearline_size) +
geom_vline(xintercept=15, col=yearline, size=yearline_size) +
geom_vline(xintercept=27, col=yearline, size=yearline_size) +
geom_vline(xintercept=39, col=yearline, size=yearline_size) +
geom_vline(xintercept=51, col=yearline, size=yearline_size)

cumulative <- ggplot_gtable(ggplot_build(x))
cumulative$layout$clip[cumulative$layout$name == "panel"] <- "off"
```

### Figure 2b: Monthly preprints per category

```r
yearlabel2 = -140
x <- ggplot(monthframe, aes(x=date, y=submissions, fill=collection)) +
geom_bar(stat="identity", color="white") +
theme_bw() +
labs(x="month", y="preprints posted (month)") +
theme(
  axis.text.x=element_blank(),
  axis.text = element_text(size=big_fontsize, color = themedarktext),
  axis.title = element_text(size=big_fontsize),
  legend.position="none"
) +
annotation_custom(
  grob = textGrob(label = "2014", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel2, ymax = yearlabel2, xmin = 3, xmax = 3) +
geom_vline(xintercept=3, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2015", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel2, ymax = yearlabel2, xmin = 15, xmax = 15) +
geom_vline(xintercept=15, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2016", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel2, ymax = yearlabel2, xmin = 27, xmax = 27) +
geom_vline(xintercept=27, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2017", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel2, ymax = yearlabel2, xmin = 39, xmax = 39) +
geom_vline(xintercept=39, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2018", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel2, ymax = yearlabel2, xmin = 51, xmax = 51) +
geom_vline(xintercept=51, col=yearline, size=yearline_size)

monthly <- ggplot_gtable(ggplot_build(x))
monthly$layout$clip[monthly$layout$name == "panel"] <- "off"
```

### Figure 2 combined

```r
legendplot <- ggplot(
    monthframe, aes(x=date, y=submissions, fill=collection)
  ) +
  geom_bar(stat="identity", color="white") +
  guides(fill=guide_legend(ncol=3)) +
  theme(legend.text = element_text(size=big_fontsize))

legend <- get_legend(legendplot)

pics <- plot_grid(cumulative, monthly,
  labels=c("(a)", "(b)"),
  rel_heights = c(1,1),
  label_x = 0, label_y = 0,
  hjust = -0.5, vjust = -0.5,
  ncol = 1, nrow = 2,
  align = "v"
)

plot_grid(
  pics, legend,
  rel_heights = c(3,1),
  ncol=1, nrow=2
)
```




## Figure 3

### Figure 3a: Publication rate by month

```sql
SELECT month, posted, published, published::decimal/posted AS rate
FROM (
  SELECT EXTRACT(YEAR FROM a.posted)||'-'||lpad(EXTRACT(MONTH FROM a.posted)::text, 2, '0') AS month,
	COUNT(a.id) AS posted,
  COUNT(p.doi) AS published
  FROM paper.articles a
  LEFT JOIN paper.article_publications p ON a.id=p.article
  GROUP BY month
  ORDER BY month
) AS counts
```

```r
paperframe = read.csv('publication_rate_month.csv')

yearlabel = -0.06

x <- ggplot(paperframe, aes(x=month, y=rate)) +
geom_point() +
labs(x = "month", y = "published") +
theme_bw() +
theme(
  axis.text.x=element_blank(),
  axis.text = element_text(size=big_fontsize, color = themedarktext),
  axis.title = element_text(size=big_fontsize),
  axis.title.y = element_text(size=big_fontsize, vjust=0)
) +
annotation_custom(
  grob = textGrob(label = "2014", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 3, xmax = 3) +
geom_vline(xintercept=3, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2015", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 15, xmax = 15) +
geom_vline(xintercept=15, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2016", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 27, xmax = 27) +
geom_vline(xintercept=27, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2017", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 39, xmax = 39) +
geom_vline(xintercept=39, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2018", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 51, xmax = 51) +
geom_vline(xintercept=51, col=yearline, size=yearline_size)

proportions <- ggplot_gtable(ggplot_build(x))
proportions$layout$clip[proportions$layout$name == "panel"] <- "off"
```


### Figure 3b: Preprint publications per journal

Consolidate duplicates:

| Original | Variant |
| --- | ----------- |
| Acta Crystallographica Section D Structural Biology | Acta Crystallographica Section D |
| Alzheimer's & Dementia | Alzheimers & Dementia |
| American Journal of Physiology - Renal Physiology | American Journal of Physiology-Renal Physiology |
| Bioinformatics | Bioinformatics (with trailing space) |
| Cell Death & Differentiation | Cell Death and Differentiation |
| Cell Death & Disease | Cell Death and Disease |
| Cell Host & Microbe | Cell Host and Microbe |
| Development | Development (Cambridge, England) |
| Epidemiology & Infection | Epidemiology and Infection |
| G3: Genes|Genomes|Genetics | G3 |
|  | G3&#58; Genes|Genomes|Genetics |
|  | G3 Genes|Genomes|Genetics |
|  | G3 (Bethesda, Md.) |
|  | Genes|Genomes|Genetics |
| Integrative Biology | Integrative Biology : Quantitative Biosciences From Nano To Macro |
|  | Integrrative Biology |
| Journal Of Physical Chemistry B | Journal Of Physical Chemistry. B |
| Methods | Methods (San Diego, Calif.) |
| Molecular & Cellular Proteomics | Molecular & Cellular Proteomics : MCP |
| Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences | Philosophical Transactions of the Royal Society A: Mathematical,				Physical and Engineering Sciences
|  | Philosophical Transactions A |
| Philosophical Transactions of the Royal Society B: Biological Sciences | Philosophical Transactions B |
| Plant & Cell Physiology | Plant and Cell Physiology |
| Plant, Cell & Environment | Plant, Cell and Environment |
| Proceedings of the Royal Society B: Biological Sciences | Proceedings B |
|   | Proceedings. Biological Sciences |
| PNAS | Proceedings of the National Academy of Sciences |
| Science | Science (New York, N.Y.) |
| SLAS Discovery | SLAS DISCOVERY: Advancing Life Sciences R&D |
| SLAS Technology | SLAS TECHNOLOGY: Translating Life Sciences Innovation |


```sql
SELECT topname AS journal, tally, REPLACE(collection, '-', ' ') AS collection
FROM (
	SELECT max(journal) AS topname, COUNT(article) AS tally, collection
	FROM (
		SELECT REGEXP_REPLACE(p.publication, '^The Journal', 'Journal') AS journal, p.article, a.collection
		FROM paper.article_publications p
		INNER JOIN paper.articles a ON p.article=a.id
	) AS stripped
	GROUP BY lower(journal), collection
	ORDER BY max(journal), tally DESC
) AS biglist
WHERE topname IN (
	SELECT journal FROM (
		SELECT max(journal) AS journal, COUNT(article) AS tally
		FROM (SELECT REGEXP_REPLACE(publication, '^The Journal', 'Journal') AS journal, article FROM paper.article_publications) AS stripped
		GROUP BY lower(journal)
		ORDER BY tally DESC, max(journal)
		LIMIT 30
	) AS ranks
)
```

```r
pubframe = read.csv('publications_per_journal_categorical.csv')

totals <- pubframe %>%
  group_by(journal) %>%
  summarize(total = sum(tally))

journals <- ggplot(pubframe, aes(x=journal, y=tally, fill=collection)) +
  geom_bar(stat="identity", color="white") +
  aes(x = reorder(journal, tally, sum), y = tally, label = tally, fill = collection) +
  coord_flip() +
  labs(y = "preprints published") +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text = element_text(size=big_fontsize, color = themedarktext),
    axis.title = element_text(size=big_fontsize),
    axis.title.y = element_blank(),
    legend.position="none"
  )
```


### Figure 3 combined

```r
main <- plot_grid(proportions, journals, labels=c("(a)", "(b)"),
  rel_heights = c(3,10),
  ncol = 1, nrow = 2,
  label_x = 0, label_y = 0,
  hjust = -0.5, vjust = -0.5
)


legendplot <- ggplot(pubframe, aes(x=journal, y=tally, fill=collection)) +
  geom_bar(stat="identity", color="white") +
  aes(x = reorder(journal, tally, sum), y = tally, label = tally, fill = collection) +
  guides(fill=guide_legend(ncol=3))

legend <- get_legend(legendplot)

plot_grid(
  main, legend,
  rel_heights = c(3,1),
  ncol=1, nrow=2
)
```

### Figure 4: Publications by category

```sql
SELECT p.collection, p.published, t.total, (p.published::decimal / t.total)
FROM (
  SELECT a.collection, COUNT(p.article) AS published
  FROM paper.article_publications p
  INNER JOIN paper.articles a ON p.article=a.id
  GROUP BY collection
  ORDER BY collection
) AS p
INNER JOIN (
  SELECT collection, COUNT(id) AS total
  FROM paper.articles
  GROUP BY collection
  ORDER BY collection
) AS t ON t.collection=p.collection
```

### Figure 4a: Total publications by category
```r
pubframe = read.csv('publications_per_category.csv')
cattotals <- ggplot(
    data=pubframe,
    aes(
      x=reorder(collection, proportion),
      y=published,
      label=published,
      fill=collection
    )
  ) +
  geom_bar(stat="identity") +
  labs(x="", y="count") +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text.x = element_blank(),
    legend.position="none",
    axis.text = element_text(size=big_fontsize, color = themedarktext)
  )
```

### Figure 4b: Publication rate by category

```r
pubframe = read.csv('publications_per_category.csv')

catproportion <- ggplot(
    data=pubframe,
    aes(
      x=reorder(collection, proportion),
      y=proportion,
      fill=collection
    )
  ) +
  geom_bar(stat="identity") +
  geom_hline(yintercept=0.40979, col=themedarkgrey, linetype="dashed", size=1) +
  annotate("text", y=0.48, x=4, label="overall proportion: 0.4098") +
  labs(y="proportion") +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    axis.text = element_text(color = themedarktext),
    # axis.title = element_text(size=big_fontsize),
    axis.text.x = element_text(size=big_fontsize, color = themedarktext, angle = 45, hjust = 1),
    axis.text.y = element_text(size=big_fontsize, color = themedarktext, hjust = 1),
    axis.title.x = element_blank(),
    legend.position="none"
  )
```

### Figure 4 combined

```r
plot_grid(cattotals, catproportion,
  labels=c("(a)", "(b)"),
  rel_heights = c(1,2),
  label_x = 0.013, label_y = 0,
  hjust = -0.5, vjust = -0.5,
  ncol = 1, nrow = 2,
  align = "v"
)
```





## Figure 5: Downloads for publications

### Figure 5a: Median bioRxiv downloads per journal

```sql
SELECT d.article, d.downloads, p.publication AS journal
FROM paper.alltime_ranks d
LEFT JOIN paper.article_publications p ON d.article=p.article
WHERE p.publication IS NULL OR lower(p.publication) IN (
	SELECT lower(journal) FROM (
		SELECT max(journal) AS journal, COUNT(article) AS tally
		FROM (SELECT REGEXP_REPLACE(publication, '^The Journal', 'Journal') AS journal, article FROM paper.article_publications) AS stripped
		GROUP BY lower(journal)
		ORDER BY tally DESC, max(journal)
		LIMIT 30
	) AS ranks
)
ORDER BY journal DESC, d.downloads DESC
```

```r
journalframe = read.csv('downloads_journal.csv')
# aggregate(downloads~journal, data=journalframe, median)

ggplot(data=journalframe, aes(
  x=reorder(journal, downloads, FUN=median),
  y=downloads
)) +
geom_boxplot(
  outlier.shape = NA, coef=0,
  fill=themepurple) +
theme_bw() +
coord_flip(ylim=c(0, 3800)) +
labs(x="journal") +
theme(
    panel.grid.major.y = element_blank(),
    axis.title = element_text(size=big_fontsize),
    axis.text= element_text(size=big_fontsize, color = themedarktext, hjust = 1),
    legend.position="none"
  )
```










## Figure S1

## Mean authors per paper

Extracting list of articles, the month and year they were posted, and how many authors they had:

```sql
SELECT
  a.id, a.collection,
  EXTRACT(MONTH FROM a.posted) AS month,
  EXTRACT(YEAR FROM a.posted) AS year,
  COUNT(w.author) AS authors
FROM paper.articles a
LEFT JOIN paper.article_authors w ON w.article=a.id
GROUP BY a.id, a.collection, a.posted
ORDER BY year, month;
```

Building figure:

```r
authorframe = read.csv('authors_per_paper.csv')

medians <- ddply(authorframe, .(collection), summarise, med = median(authors))

ggplot(data=authorframe, aes(
  x=reorder(collection, authors, FUN=median),
  y=authors,
  fill=collection
)) +
geom_boxplot(outlier.shape = NA) +
scale_y_continuous() +
coord_flip(ylim=c(0,25)) +
theme_bw() +
labs(x="collection", y="authors per paper") +
theme(
  panel.grid.major.y = element_blank(),
  legend.position="none",
  axis.text = element_text(size=big_fontsize, color = themedarktext),
  axis.title.y = element_text(size=big_fontsize),
) +
# geom_hline(yintercept=median(authorframe$authors), col=themedarkgrey, linetype="dashed", size=1) +
# annotate("text", y=median(authorframe$authors)+4, x=1, label=paste("overall median:", round(median(authorframe$authors), 2))) +
```


### Figure S2: Monthly downloads overall

```r
monthframe=read.csv('downloads_per_month_cumulative.csv')

yearlabel = -100000

x <- ggplot(monthframe, aes(x=date, y=month, group=collection, fill=collection)) +
geom_bar(stat="identity", color="white") +
labs(x = "month", y = "total downloads") +
theme_bw() +
scale_y_continuous(breaks=seq(0, 1250000, 250000), labels=comma) +
theme(
  axis.text.x=element_blank(),
  axis.text.y = element_text(size=big_fontsize, color = themedarktext),
  axis.title.y = element_text(size=big_fontsize),
  legend.text = element_text(size=big_fontsize),
  legend.position = "bottom"
) +
annotation_custom(
  grob = textGrob(label = "2014", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 3, xmax = 3) +
geom_vline(xintercept=3, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2015", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 15, xmax = 15) +
geom_vline(xintercept=15, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2016", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 27, xmax = 27) +
geom_vline(xintercept=27, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2017", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 39, xmax = 39) +
geom_vline(xintercept=39, col=yearline, size=yearline_size) +
annotation_custom(
  grob = textGrob(label = "2018", hjust = 1, vjust=1, gp = gpar(fontsize = big_fontsize, col=themedarktext)),
  ymin = yearlabel, ymax = yearlabel, xmin = 51, xmax = 51) +
geom_vline(xintercept=51, col=yearline, size=yearline_size) +
guides(fill=guide_legend(ncol=3))

main <- ggplot_gtable(ggplot_build(x))
main$layout$clip[main$layout$name == "panel"] <- "off"
grid.draw(main)

legendplot <- ggplot(monthframe, aes(x=date, y=month, group=collection, fill=collection)) +
  geom_bar(stat="identity", color="white") +


legend <- get_legend(legendplot)
plot_grid(
  main, legend,
  rel_heights = c(3,1),
  ncol=1, nrow=2
)
```

## Figure S3: Median downloads per year

```sql
SELECT d.article, d.downloads, EXTRACT(YEAR FROM a.posted) AS year
FROM paper.alltime_ranks d
INNER JOIN paper.articles a ON d.article=a.id
WHERE a.collection IS NOT NULL;
```

```r
paperframe = read.csv('downloads_per_year.csv')
mediandownloads <- aggregate(downloads~year,data=paperframe,median)

# Stacked density plot:
ggplot(data=paperframe, aes(
    x=downloads, group=year, fill=year
  )) +
  geom_density() +
  scale_x_continuous(trans="log10", labels=comma) +
  scale_y_continuous(breaks=seq(0.5, 1.5, 0.5)) +
  coord_cartesian(xlim=c(8,30000)) +
  theme_bw() +
  labs(x="total downloads", y="probability density") +
  theme(
    legend.position="none",
    axis.text = element_text(size=big_fontsize),
    axis.title = element_text(size=big_fontsize)
  ) +
  geom_vline(aes(xintercept=downloads), data=mediandownloads,
    col=themeorange, linetype="dashed", size=1) +
  geom_text(
    aes(label = paste("median:", downloads), x=downloads*3.5, y=1.2),
    data=mediandownloads
  ) +
  facet_grid(rows = vars(year)) +
  theme(
    strip.text = element_text(size=big_fontsize)
  )

leveneTest(downloads~year, data=paperframe)
kruskal.test(downloads~year, data=paperframe)
library(FSA)
dunnTest(downloads~year, data=paperframe, method="bh")
```

## Figure S4

# Downloads in first month on bioRxiv
```sql
SELECT a.id, t.month, t.year, t.pdf AS downloads
FROM paper.articles a
LEFT JOIN paper.article_traffic t
  ON a.id=t.article
  AND t.year = (
    SELECT MIN(year)
    FROM paper.article_traffic t
    WHERE t.article = a.id
  )
  AND t.month = (
    SELECT MIN(month)
    FROM paper.article_traffic t
    WHERE a.id=t.article AND
      year = (
        SELECT MIN(year)
        FROM paper.article_traffic t
        WHERE t.article = a.id
      )
  )
ORDER BY id
```

# Best month of downloads
```sql
SELECT a.id, EXTRACT(year FROM a.posted) AS year, t.pdf AS downloads
FROM paper.articles a
LEFT JOIN paper.article_traffic t
  ON a.id=t.article
  AND t.pdf = (
    SELECT MAX(pdf)
    FROM paper.article_traffic t
    WHERE t.article = a.id
  )
ORDER BY id
```


```r
firstframe = read.csv('downloads_by_first_month.csv')
maxframe = read.csv('downloads_max_by_year_posted.csv')

firstplot <- ggplot(data=firstframe, aes(
  x=year,
  y=downloads,
  group=year
)) +
geom_boxplot(outlier.shape = NA, coef=0) +
scale_y_continuous(labels=comma) +
coord_cartesian(ylim=c(0,250)) +
theme_bw() +
labs(x="year", y="downloads in first month") +
theme(
  legend.position="none",
  axis.text.y = element_text(size=big_fontsize, hjust=0.9),
  axis.text.x = element_blank()
)

maxplot <- ggplot(data=maxframe, aes(
  x=year,
  y=downloads,
  group=year
)) +
geom_boxplot(outlier.shape = NA, coef=0) +
scale_y_continuous(labels=comma) +
coord_cartesian(ylim=c(0,250)) +
theme_bw() +
labs(x="year", y="downloads in best month") +
theme(
  legend.position="none",
  axis.text = element_text(size=big_fontsize, hjust=0.9)
)

plot_grid(firstplot, maxplot, labels=c("(a)", "(b)"),
  label_x = 0, label_y = 0,
  hjust = -0.5, vjust = -0.5,
  ncol = 1, nrow = 2,
  align = "v"
)
```