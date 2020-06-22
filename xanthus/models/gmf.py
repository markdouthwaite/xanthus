from tensorflow.keras import Model

from xanthus.models import utils, base


class GeneralizedMatrixFactorizationModel(base.NeuralRecommenderModel):
    def _build_model(self, dataset, **kwargs) -> Model:
        return utils.get_mf_model(
            dataset.all_users.shape[0],
            dataset.all_items.shape[0],
            **self._config,
            reg=1e-6,
        )
