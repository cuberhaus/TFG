from clases.model_utils import parse_model_filename

file="FasterRCNN_BATCH_SIZE-2_LR-0.0016943510707123216_WEIGHT_DECAY-0.0003878798725989378_NUM_EPOCHS-7_epoch-6_20231128055328"

parsed = parse_model_filename(file)
print(parsed)