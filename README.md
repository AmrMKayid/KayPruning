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
- [Pruning Makes Faster and Smaller Neural Networks | Two Minute Papers](https://www.youtube.com/watch?v=3yOZxmlBG3Y)
- [Pruning deep neural networks to make them fast and small](https://jacobgil.github.io/deeplearning/pruning-deep-learning)
- [Neural Network Pruning](http://primo.ai/index.php?title=Neural_Network_Pruning)
- [Awesome-Pruning](https://github.com/he-y/Awesome-Pruning)
- [Effective TensorFlow 2.0](https://www.tensorflow.org/beta/guide/effective_tf2)


### Contributors:

| [<img src="https://avatars0.githubusercontent.com/u/18689888" width="150px;" height="150px;"/><br /><sub><b>Amr M. Kayid</b></sub>](https://github.com/AmrMKayid)|
| :---: | 


