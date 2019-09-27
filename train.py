from kaypruning.data import DataBunch
from kaypruning.models import *
from kaypruning.trainer import Trainer
from kaypruning.utils import *

if __name__ == '__main__':
    db = DataBunch('mnist')
    model = get_model('BaseModel')

    glogger.info('Training')
    trainer = Trainer(model=model, db=db, epochs=1)
    trainer.run()
    glogger.info(trainer.metrics)

    glogger.info('Pruning')
    trainer.run_pruning()
    glogger.info(trainer.print_metrics())

    glogger.info('Plotting')
    w_loss, w_acc, u_loss, u_acc = trainer.get_metrics()

    # %%
    plot(y=w_loss, prune_type='Weight', type='Loss')
    # %%
    plot(y=w_acc, prune_type='Weight', type='Accuracy')
    # %%
    plot(y=u_loss, prune_type='Unit', type='Loss')
    # %%
    plot(y=u_acc, prune_type='Unit', type='Accuracy')
    # %%
    plt.plot(prune_configs.k, w_acc, label="Weight Pruning")
    plt.plot(prune_configs.k, u_acc, label="Unit Pruning")

    plt.xlabel('Sparsity (%)')
    plt.ylabel('Test Accuracy')
    plt.title('Pruning')

    plt.legend()
    plt.show()
