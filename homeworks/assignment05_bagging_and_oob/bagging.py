import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            self.indices_list.append(np.random.randint(0, data_length, size=data_length))
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag, target_bag = data[self.indices_list[bag]], target[self.indices_list[bag]]
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''
        return sum(model.predict(data) for model in self.models_list) / len(self.models_list)
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # indices_list contains bootstraps
        # if not i in indices_list[j]: # i is not in bootstrap j
        for i in range(len(self.data)):
            oob_models = [self.models_list[j] for j in range(len(self.models_list)) if not i in self.indices_list[j]]
            list_of_predictions_lists[i] = np.array([model.predict([self.data[i]]) for model in oob_models])
            # prediction for self.data[i] from all models that didnt see it
        # print(list_of_predictions_lists)
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        #print(self.list_of_predictions_lists)
        self.oob_predictions = [None if pred is None else np.mean(pred) for pred in self.list_of_predictions_lists]
        # print(len(self.list_of_predictions_lists),max(len(self.list_of_predictions_lists[i]) for i in range(len(self.list_of_predictions_lists))), len(self.oob_predictions))
        # print(np.nanmean(self.list_of_predictions_lists[0]))
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        self._get_averaged_oob_predictions()
        # assert len(self.target) > 0
        # assert len(self.oob_predictions) > 0
        return np.nanmean((self.target - self.oob_predictions) ** 2)
