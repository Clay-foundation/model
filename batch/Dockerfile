# List of aws images https://github.com/aws/deep-learning-containers/blob/master/available_images.md
FROM mcr.microsoft.com/planetary-computer/python

# Fetch-and-run setup.
# https://aws.amazon.com/blogs/compute/creating-a-simple-fetch-and-run-aws-batch-job/
ADD fetch_and_run.sh /usr/local/bin/fetch_and_run.sh
WORKDIR /tmp
ENTRYPOINT ["/usr/local/bin/fetch_and_run.sh"]
