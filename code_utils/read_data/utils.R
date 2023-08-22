#' @export
#' @importFrom Matrix sparseMatrix colSums
#' @importFrom SingleCellExperiment SingleCellExperiment counts
#' @importFrom S4Vectors metadata metadata<-
#' @importFrom rhdf5 h5read
#' @rdname readVisium
read10Xh5 <- function(dirname, fname = "filtered_feature_bc_matrix.h5") {
  spatial_dir <- file.path(dirname, "spatial")
  h5_file <- file.path(dirname, fname)
  
  if (!dir.exists(spatial_dir)) {
    stop("Spatial directory does not exist:\n  ", spatial_dir)
  }
  
  if (!file.exists(h5_file)) {
    stop("H5 file does not exist:\n  ", h5_file)
  }
  
  colData <- .read_spot_pos(spatial_dir)
  
  non.zero.indices <- .extract_indices(
    h5read(h5_file, "matrix/indices"),
    h5read(h5_file, "matrix/indptr")
  )
  
  rowData <- h5read(h5_file, "matrix/features/id")
  
  .counts <- sparseMatrix(
    i = non.zero.indices$i,
    j = non.zero.indices$j,
    x = h5read(h5_file, "matrix/data"),
    dims = h5read(h5_file, "matrix/shape"),
    dimnames = list(
      rowData,
      h5read(h5_file, "matrix/barcodes")
    ),
    index1 = FALSE
  )
  .counts <- .counts[, rownames(colData)]
  
  sce <- SingleCellExperiment(
    assays = list(
      counts = .counts
    ),
    rowData = rowData,
    colData = colData
  )
  
  # Remove spots with no reads for all genes.
  sce <- sce[, Matrix::colSums(counts(sce)) > 0]
  
  sce
}

#' Load spot positions.
#'
#' @param dirname Path to spaceranger outputs of spatial pipeline, i.e., "outs/spatial".
#'     This directory must contain a file for the spot positions at
#'     \code{tissue_positions_list.csv} (before Space Ranger V2.0) or
#'     \code{tissue_positions.csv} (since Space Ranger V2.0).
#'
#' @return Data frame of spot positions.
#'
#' @keywords internal
#'
#' @importFrom utils read.csv
#' @importFrom tibble as_tibble
#' @importFrom dplyr inner_join
#' @importFrom tidyr uncount
.read_spot_pos <- function(dirname, barcodes = NULL) {
  if (file.exists(file.path(dirname, "tissue_positions_list.csv"))) {
    message("Inferred Space Ranger version < V2.0")
    colData <- read.csv(file.path(dirname, "tissue_positions_list.csv"), header = FALSE)
    colnames(colData) <- c("barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres")
  } else if (file.exists(file.path(dirname, "tissue_positions.csv"))) {
    message("Inferred Space Ranger version >= V2.0")
    colData <- read.csv(file.path(dirname, "tissue_positions.csv"))
  } else {
    stop("No file for spot positions found in ", dirname)
  }
  
  if (!is.null(barcodes)) {
    colData <- inner_join(
      colData,
      barcodes,
      by = c("barcode" = "V1")
    )
  }
  
  rownames(colData) <- colData$barcode
  colData <- colData[colData$in_tissue > 0, ]
  return(colData)
}

#' Extract row and column indices of the count matrix from h5 file.
#'
#' @param idx Row index of corresponding element in the non-zero count matrix.
#' @param new.start Index of the start of each column corresponding to
#'     \code{idx} and the non-zero count matrix.
#' @param zero.based Whether the \code{} and \code{} are zero-based or not.
#'     (By default is TRUE)
#'
#' @return List of row (i) and column (j) indices of the non-zero elements
#'     in the count matrix.
#'
#' @keywords internal
#'
#' @importFrom tibble as_tibble
#' @importFrom tidyr uncount
.extract_indices <- function(idx, new.start, zero.based = TRUE) {
  if (length(idx) < 1) {
    return(NULL)
  }
  
  idx.cnts <- do.call(
    rbind,
    lapply(
      seq_len(length(new.start))[-1],
      function(x) c(x - ifelse(zero.based, 2, 1), new.start[[x]] - new.start[[x - 1]])
    )
  )
  colnames(idx.cnts) <- c("id", "n")
  
  return(
    list(
      i = idx,
      j = as.integer(uncount(as_tibble(idx.cnts), n)[[1]]),
      new.start = new.start
    )
  )
}
#' @export
#' @importFrom RCurl url.exists
#' @importFrom utils download.file
#' @importFrom assertthat assert_that
#' @importFrom BiocFileCache BiocFileCache bfcrpath
getRDS <- function(dataset, sample, cache = TRUE) {
  url <- "https://fh-pi-gottardo-r-eco-public.s3.amazonaws.com/SpatialTranscriptomes/%s/%s.rds"
  url <- sprintf(url, dataset, sample)
  assert_that(url.exists(url), msg = "Dataset/sample not available")
  
  if (cache) {
    bfc <- BiocFileCache()
    local.path <- bfcrpath(bfc, url)
  } else {
    local.path <- tempfile(fileext = ".rds")
    download.file(url, local.path, quiet = TRUE, mode = "wb")
  }
  
  ret <- readRDS(local.path)
  
  # Rename columns of colData of `ret` for compatibility reasons.
  if (any(c("row", "col") %in% colnames(colData(ret)))) {
    col.names <- colnames(colData(ret))
    col.names <- gsub("row", "array_row", col.names)
    col.names <- gsub("col", "array_col", col.names)
    colnames(colData(ret)) <- col.names
  }
  
  ret
}
