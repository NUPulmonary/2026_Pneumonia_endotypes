library(DESeq2)
library(vsn)


options(error = function() {traceback(3); q(1)})

BASE <- '/projects/b1196/ewa_group/serniczek/data/05_pseudobulk/01_sample_pathways'
DIR <- '/projects/b1196/ewa_group/serniczek/data/05_pseudobulk/10_vap_cured'


sanitize_name <- function(name) {
  name <- gsub(' ', '_', name, fixed = TRUE)
  name <- gsub('*', '', name, fixed = TRUE)
  name <- gsub(';', '_and', name, fixed = TRUE)
  name <- gsub('/', '_', name, fixed = TRUE)
  return(name)
}


process_cell_type <- function(meta_path, labels_path, count_path, output_path) {
  cell_type_dir <- dirname(meta_path)
  cell_type <- basename(cell_type_dir)
  task <- basename(dirname(labels_path))
  writeLines(sprintf("%s]\tStarting %s", task, basename(cell_type_dir)))

  sample_groups <- read.csv(labels_path)
  meta <- read.csv(meta_path, row.names = 1)
  meta <- merge(meta, sample_groups, by.x = 'sample', by.y = 'bal_barcode', all.x = TRUE)
  meta$group <- as.character(meta$group)
  meta$group[is.na(meta$group)] <- 'discard'

  save_dir <- output_path
  R.utils::mkdirs(save_dir)

  if (length(unique(meta$sex)) < 2) {
    return()
  }

  groups_with_3_or_more_samples <- 0
  for (group in unique(sample_groups$group)) {
    if (sum(meta$group == group) > 2) {
      groups_with_3_or_more_samples <- groups_with_3_or_more_samples + 1
    }
  }
  if (groups_with_3_or_more_samples < 2) {
    return()
  }

  write.csv(meta, sprintf('%s/%s', save_dir, 'meta.csv'))

  counts <- read.csv(count_path, sep='\t', check.names = FALSE, row.names = 1)
  counts <- counts[, meta$sample]

  dds <- DESeqDataSetFromMatrix(counts, colData = meta, design = ~ group + sex)
  dds <- DESeq(dds, fitType = "local")

  pdf(sprintf('%s/%s', save_dir, 'disp-local.pdf'), width = 6, height = 4)
  plotDispEsts(dds)
  dev.off()

  # print(comparison)
  degs <- as.data.frame(results(
    dds,
    contrast = c('group', '1', '0'),
    alpha = 0.05
  ))
  write.csv(degs, sprintf('%s/%s', save_dir, 'degs.csv'))
}


args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop("Usage: Rscript 11_vap_cured_deseq2.R --meta <m> --count <c> --labels <l> --output <o>")
}

for (i in seq(1, length(args), 2)) {
  arg <- args[i]
  val <- args[i + 1]
  if (arg == "--meta") {
    meta_path <- val
  } else if (arg == "--count") {
    count_path <- val
  } else if (arg == "--labels") {
    labels_path <- val
  } else if (arg == "--output") {
    output_path <- val
  } else {
    stop(sprintf("Unknown argument: %s", arg))
  }
}

process_cell_type(meta_path, labels_path, count_path, output_path)
