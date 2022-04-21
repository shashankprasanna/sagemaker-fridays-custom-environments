eksctl create cluster dl-env \
                      --version 1.22 \
                      --nodes 2 \
                      --node-type=p3.2xlarge \
                      --timeout=40m \
                      --ssh-access \
                      --ssh-public-key shshnkp-oregon \
                      --region us-west-2 \
                      --zones=us-west-2a,us-west-2b,us-west-2c \
                      --auto-kubeconfig