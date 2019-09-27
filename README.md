<h1 align=center> KayPruning </h1>
<h2 align=center> Making Neural Networks smaller and faster!</h2>


### Getting started
It is recommended to use a virtual environment (conda).

Using **conda**:
`
conda env create --name envname --file=env.yml
`

Using **pip**:
`
pip install -r requirements.txtinst
`


### Basic usage:
Check **_forai.ipynb_** notbook

#### Simple training & pruning
```python
db = DataBunch('mnist')
model = get_model('BaseModel')

glogger.info('Training')
trainer = Trainer(model=model, db=db, epochs=1)
trainer.run()
glogger.info(trainer.metrics)

glogger.info('Pruning')
trainer.run_pruning()
glogger.info(trainer.print_metrics())
```

#### Hyper-parameters & configurations:
The configurations and hyper-parameters can be found 
in the **_configs_** package and you can change and adjust them.

### Resource:
- [J. Frankle & M. Carbin: The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks](https://www.youtube.com/watch?v=s7DqRZVvRiQ)
- [Pruning Makes Faster and Smaller Neural Networks | Two Minute Papers](https://www.youtube.com/watch?v=3yOZxmlBG3Y)
- [Pruning deep neural networks to make them fast and small](https://jacobgil.github.io/deeplearning/pruning-deep-learning)
- [Neural Network Pruning](http://primo.ai/index.php?title=Neural_Network_Pruning)
- [Awesome-Pruning](https://github.com/he-y/Awesome-Pruning)
- [Effective TensorFlow 2.0](https://www.tensorflow.org/beta/guide/effective_tf2)

### What I did/learn? 📚👨🏻‍💻
- This was my first time to learn about **_Pruning Neural Networks_**!
- In this project I decided to use **Tensorflow 2.0** to learn and have actual project using tf2.
- I tried to follow _tensorflow best practices_ and have a modularize code which can be extended to support more types of pruning and models.
- I followed some of [FOR.ai/rl](https://github.com/for-ai/rl) library design ;)
- I tried to implement everything with customization and also used Keras APIs to integrate it with the project.
- I read and watched some useful resources that helped me during the project
- I really enjoyed working on this project! 😄 👌


### Contributors:

| [<img src="https://avatars0.githubusercontent.com/u/18689888" width="150px;" height="150px;"/><br /><sub><b>Amr M. Kayid</b></sub>](https://github.com/AmrMKayid)|
| :---: | 


