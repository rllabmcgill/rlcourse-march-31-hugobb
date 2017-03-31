Deep Q-Learning
==================

This is a implementation of the paper [Human Level-Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html).

To train a model run:

`python train.py [environement] [output_dir]`

replace [environement] by one of the openai gym environement for example 'SpaceInvaders'

To test a pretrained model run:

`python test.py [environement] [params_filename]`

To visualize the training curves:

`python plot_results.py [results_path]`

**Warning**: By default it takes around 8GB of memory to run, you can change this by reducing the replay memory size with the --mem_size option.
