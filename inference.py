import os
import argparse

def main():
    """
    The main function of your running scripts. 
    """
    # default data folder
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, nargs='?', default='/input', help='input directory')
    parser.add_argument('--output', type=str, nargs='?', default='/output', help='output directory')
    args = parser.parse_args()

    ## functions are not real python functions, but are examples here.

    ## Read in your trained model
    trained_model_weights_dir = './model_weights'
    model = load_model_weights(trained_model_weights_dir)

    ## Make your prediction segmentation files
    segmentation_outputs = inference(model, args.input)

    ## Write your prediction to the output folder
    write_outputs(segmentation_outputs, args.output)

if __name__ == "__main__":
	main()