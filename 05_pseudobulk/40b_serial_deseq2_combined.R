library(DESeq2)
library(vsn)


options(error = function() {traceback(3); q(1)})


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
  writeLines(sprintf("%s]\tStarting %s", task, cell_type))

  sample_groups <- read.csv(labels_path)

  meta <- read.csv(meta_path, row.names = 1)
  meta <- merge(meta, sample_groups, by.x = 'sample', by.y = 'bal_barcode', all.x = TRUE)

  meta$group <- as.character(meta$group)
  meta <- meta[!is.na(meta$group), ]
  meta$pair_id <- as.character(meta$pair_id)

  for (group in unique(meta$group)) {
    pairs <- unique(meta$pair_id[meta$group == group])
    for (i in 1:length(pairs)) {
      new_name <- paste0('pair', i)
      meta$pair_id[meta$pair_id == pairs[i]] <- new_name
    }
  }
  meta$pair_id <- as.factor(meta$pair_id)

  save_dir <- output_path
  R.utils::mkdirs(save_dir)
  if (length(unique(meta$timepoint)) < 2) {
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

  model_matrix <- model.matrix(~ group + group:timepoint + group:pair_id, meta)
  all.zero <- apply(model_matrix, 2, function(x) all(x==0))
  idx <- which(all.zero)
  model_matrix <- model_matrix[,-idx]

  write.csv(meta, sprintf('%s/%s', save_dir, 'meta.csv'))

  counts <- read.csv(count_path, sep='\t', check.names = FALSE, row.names = 1)
  counts <- counts[, meta$sample]

  dds <- DESeqDataSetFromMatrix(counts, colData = meta, design = model_matrix)
  dds <- DESeq(dds, fitType = "local")

  print(resultsNames(dds))

  pdf(sprintf('%s/%s', save_dir, 'disp-local.pdf'), width = 6, height = 4)
  plotDispEsts(dds)
  dev.off()

  # print(comparison)
  degs <- as.data.frame(results(
    dds,
    name = 'groupvap.timepointsecond',
    alpha = 0.05,
    test = 'Wald'
  ))
  write.csv(degs, sprintf('%s/%s', save_dir, 'degs-vap.csv'))

  degs <- as.data.frame(results(
    dds,
    name = 'groupno_vap.timepointsecond',
    alpha = 0.05,
    test = 'Wald'
  ))
  write.csv(degs, sprintf('%s/%s', save_dir, 'degs-no-vap.csv'))

  degs <- as.data.frame(results(
    dds,
    contrast = list('groupvap.timepointsecond', 'groupno_vap.timepointsecond'),
    alpha = 0.05,
    test = 'Wald'
  ))
  write.csv(degs, sprintf('%s/%s', save_dir, 'degs.csv'))

  degs <- as.data.frame(results(
    dds,
    contrast = list('groupvap', 'groupno_vap'),
    alpha = 0.05,
    test = 'Wald'
  ))
  write.csv(degs, sprintf('%s/%s', save_dir, 'degs-baseline.csv'))
}



args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 8) {
  stop("Usage: Rscript 40b_serial_deseq2.R --meta <m> --count <c> --labels <l> --output <o>")
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
