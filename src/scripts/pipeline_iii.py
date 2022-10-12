import argparse
import yaml


def main(config_path):
    
    ### Loading params
    with open(config_path) as config_file:
        config = yaml.safe_load(config_file)
    


if __name__ == "__main__":

    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    main(config_path=args.config)

