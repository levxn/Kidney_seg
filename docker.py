import os
from test import predict_and_save

def main():

    input_dir = "/input"
    output_dir = "/output"
    
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The input directory '{input_dir}' does not exist.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"The output directory '{output_dir}' does not exist.")
    
    predict_and_save(input_dir, output_dir)
    print("SUCCESS!!\n")
    
if __name__ == "__main__":
    main()
