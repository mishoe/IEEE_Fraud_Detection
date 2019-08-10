class XgboostOptimizer:
    '''
    Austin Mishoe
    This is class serves as hyper-parameter optimizer for xgboost.  Use the update_params() method to
    pass in a dictionary of parameters to optimize.
    When updating parameters, input types of list, 1D np.arrays are allowed.
    There are three main search methodologies available in this class:
    1) In the fit_seq_search() function, the search methodology is a sequential search that looks to the right and left of a
    particular value and train until the evaluation metric is over fit, or fails to improve.
    2) In the fit_random_search() function, the search methodology is a random search that looks to sample values for each
    of the non-regularizing parameters and keeps the parameter values that optimize the evaluation metric w/o overfitting
    3) In the fit_full_search() function, the random search is first applied, and then the sequential search is applied
    Use set_initial_xgb_params(x.self.xgb_params_best) to set the initial parameters of the search based on prior
    searches
    Available parameters and their defaults{
    'num_boost_rounds': 25,
     'final_num_boost_rounds': 100,
     'max_over_fit': 0.03,
     'maximize': True,
     'max_mins_per_param': 2,
     'max_minutes_total':25,
     'build_past': 2,
     'use_r2': True,
     'early_stopping_rounds': 5}
     'max_depth': range(3, 10),
     'min_child_weight': range(50, 1000, 25),
     'gamma': range( 0, 100, 5),
     'eta': np.arange(.01, .4, .01),
     'colsample_bytree': [1, .9, .8],
     'lambda': np.arange(1, 0, -.01),
     'alpha': np.arange(0, 1, .01),
     'max_delta_step': range(0, 10, 1),
     'objective': 'reg:linear',
     'silent':1,
     'eval_metric': 'rmse'}
     #Pointers do exists in python... specifically when assigning a dictionary, modifying the new one, it modifies the old.
     #example of the pointer that was wreaking havoc on the code:
     dict1 = {'first':'hello', 'second':'world'}
     dict2 = dict1 # pointer assignation mechanism
     dict2['first'] = 'bye'
     dict1
     '''

    def __init__(self,verbose=True):
        import numpy as np
        self.verbose = verbose
        self.train_params = {'num_boost_rounds': 100,
                             'max_over_fit': .05,
                             'maximize': True,
                             'max_mins_per_param': 5,
                             'max_minutes_total': 25,
                             'build_past': 2,
                             'use_r2': True,
                             'early_stopping_rounds': 2,
                             'num_rand_samples': 100,
                             'xgb_model': None}  # this parameter is used for iterative training

        self.xgb_params = {'max_depth': range(3, 12),
                             'min_child_weight': range(50, 1000, 25),
                             'gamma': range( 0, 100, 5),
                             'eta': np.arange(.01, .25, .01),
                             'colsample_bytree': np.arange(1, .75, -.05),
                             'lambda': np.arange(1, 0, -.02),
                             'alpha': np.arange(0, 1, .02),
                             'objective': 'reg:linear',
                             'silent':1,
                             'eval_metric': 'rmse'}

        self.random_search_params = ['max_depth','min_child_weight','eta']
        self.evaluation_params = ['objective', 'eval_metric', 'silent']

        self.descending_params = ['lambda','colsample_bytree'] #desc_params is used in the logic of set_init_params method
        self.user_set_initial_params = {} #keeps track of which initial param values the user has declared
        self.xgb_params_start = {}
        self.xgb_params_best = {}
        self.set_initial_xgb_params()
        self.best_score = {'train': 0, 'eval': 0}
        self.n_models_tried = 0
        self.best_model = None
        self.base_line_model = None
        self.train_results = None
        self.eval_results = None
        self.base_score = None
        self.overfit_metric = None #string passed with which metric to use in overfitting computation
        self.tune_results = {'train': [], 'eval': [], 'param': [], 'val': [], 'temp_params': []}
        self.order = ['max_depth', 'min_child_weight', 'gamma', 'eta', 'lambda', 'alpha']

    def update_params(self, params):
        '''
        a Function to update parameters for the xgboost_optimizer
        xgb_params: dictionary of parameters passed as a diction to xgboost
            Note: this may include lists ie: {'eta': [.1, .2]}.
        This function updates the parameter grid used to search in parameter tuning
        train_params: args passed into the xgb.train function
        :param params: a dictionary of parameters
        :return:Void
        '''
        for key in params.keys():
            if key in self.train_params.keys():
                self.train_params[key] = params[key]
            else:
                self.xgb_params[key] = params[key]
        # call set_init_params which then extracts initial values and calls split_grid()
        self.set_initial_xgb_params()
        #self.set_random_param_range()

    def set_initial_xgb_params(self,initial_params={}):
        '''
        This class is used to initialize the starting values of the parameters in the param tuning process.
        The default values used are in the __init__ function, any parameters passed in will be used instead
        of the default ones
        :param initial_params: dictionary of initial values for the parameters to be used in the parameter tuning
        :return: Void
        '''
        import numpy
        for key in self.xgb_params.keys():
            if key not in self.evaluation_params:
                # flatten param list to 1D list if necessary, o.w. use the current param range
                if type(self.xgb_params[key]) == list and type(self.xgb_params[key][0]) == list:
                    tmp_vals = [item for sublist in self.xgb_params[key] for item in sublist]
                    # due to the fact that the flattening leaves params out of order, the sorting is necessary
                    if key in self.descending_params:
                        tmp_vals.sort(reverse=True)
                    else:
                        tmp_vals.sort(reverse=False)
                else:
                    tmp_vals = self.xgb_params[key]
                # logic to set the initial starting value for the sequential search
                if initial_params.get(key) == None and self.user_set_initial_params.get(key)==None:
                    #check if the param values is a range. if range, pull first value
                    if type(tmp_vals) in [list, range] or type(tmp_vals).__module__ == numpy.__name__:
                        self.xgb_params_start[key] = tmp_vals[0]
                    else:
                        self.xgb_params_start[key] = tmp_vals
                elif initial_params.get(key) != None:
                    self.user_set_initial_params[key]= initial_params.get(key)
                    self.xgb_params_start[key] = initial_params.get(key)
                else:
                    self.xgb_params_start[key] = self.user_set_initial_params[key]

        #set objective,eval_metric, silent params
        for param in self.evaluation_params:
            self.xgb_params_start[param] = self.xgb_params[param]

        self.__split_grid()

    def __split_grid(self):
        '''
        This is an internal function that is called from set_initial_xgb_params and update_params
        This function modifies xgb_params to separate the search list for each parameter into two.
        This allows searching on both the left and right side of the initial parameter value
        :return:Void
        '''
        for key in self.xgb_params.keys():
            if key not in self.evaluation_params:
                # flatten param list to 1D list if necessary (may be a list of list from left-right coersion
                if type(self.xgb_params[key]) == list and type(self.xgb_params[key][0]) == list:
                    self.xgb_params[key] = [item for sublist in self.xgb_params[key] for item in sublist]
                # create a list if the datatype is non-iterable (occurs when single value is passed)
                elif type(self.xgb_params[key]) in [int, float, str]:
                    self.xgb_params[key] = [self.xgb_params[key]]
                # find vals to the left of the base_param value
                left_vals = [val for val in self.xgb_params[key] if val < self.xgb_params_start.get(key)]
                left_vals.sort(reverse=True)
                # find vals to the right of the base_param value
                right_vals = [val for val in self.xgb_params[key] if val >= self.xgb_params_start.get(key)]
                right_vals.sort(reverse=False)
                # assign the param array for the param search
                if len(left_vals) != 0 and len(right_vals) != 0:
                    self.xgb_params[key] = [left_vals, right_vals]
                elif len(left_vals) != 0:
                    self.xgb_params[key] = left_vals
                elif len(right_vals) != 0:
                    self.xgb_params[key] = right_vals
                else:
                    self.xgb_params[key] = None

    def __variable_not_set(self,variable):
        '''
        returns True if variable is equal to None, {},[]
        :param variable: any python variable
        :return: (bool) True if variable unasigned
        '''
        if bool(variable):
            return False
        else:
            return True

    def __set_feval_model_metric(self):
        '''
        Use this method to create any custom evaluation metrics for use in xgboost training
        This is an internal function that is called within the fit methods.
        This method assigns feval(custom eval function for training) based
        on the type of model/if r2 is chosen to be used and assigns the metric
        used in the overfitting computation.
        :return: Void
        '''
        from sklearn.metrics import r2_score
        # Assign feval to r2 if the model is a regression and user input chooses to use as a metric
        if self.xgb_params.get('objective').__contains__('reg') and self.train_params['use_r2']:
            def eval_r2(preds, dtrain):
                labels = dtrain.get_label()
                return 'r2', r2_score(labels, preds)
            self.feval = eval_r2
        else:
            self.feval = None

        if self.overfit_metric==None:
            if self.xgb_params.get('objective').__contains__('reg'):
                if self.train_params['use_r2']:
                    self.overfit_metric='r2'
                else:
                    self.overfit_metric='rmse'
            elif self.xgb_params.get('objective').__contains__('multi'):
                self.overfit_metric = 'mlogloss'
            else:
                self.overfit_metric = 'auc'

    def fit_full_search(self, dtrain, evals):
        '''
        This method first calls fit_rand_search and then fit_seq_search to first perform a full random search
        of the parameter space, and then a sequential search to fine tune model performance
        :param dtrain: xgboost D Matrix with the labels set for traning data
        :param evals: example  evals=[(train, 'Train'), (test, "Test")]) where train and test are xgb D matrices
        :return: returns a fitted object, with best model and scores stored
        '''
        self.fit_random_search(dtrain,evals)
        # Extract the best parameters from the random search and set best params
        # as starting values for the sequential search method
        if bool(self.xgb_params_best):
            self.set_initial_xgb_params(self.xgb_params_best)
        self.fit_seq_search(dtrain, evals)

    def fit_seq_search(self, dtrain, evals):
        '''
        This method uses a heuristic grid sequence search approach to solve for the optimal parameters of a model.
        Modify all options of this methods through the attributes of the class
        :param dtrain: xgboost D Matrix with the labels set for traning data
        :param evals: example  evals=[(train, 'Train'), (test, "Test")]) where train and test are xgb D matrices
        :return: returns a fitted object, with best model and scores stored
        '''
        import xgboost
        import time
        import numpy as np

        #set custom eval metric based on params/objective passed in
        self.__set_feval_model_metric()

        if self.__variable_not_set(self.best_model):
            # if the optimization is multiclass, set num_class
            if self.xgb_params_start.get('objective').__contains__('multi'):
                self.xgb_params_start['num_class']=len(np.unique(dtrain.get_label()))

            # Build preliminary model
            model = xgboost.train(params=self.xgb_params_start,
                                 dtrain=dtrain,
                                 evals=evals,
                                 num_boost_round=self.train_params['num_boost_rounds'],
                                 early_stopping_rounds=self.train_params['early_stopping_rounds'],
                                 maximize=self.train_params['maximize'],
                                 feval=self.feval,
                                 verbose_eval=False,
                                xgb_model=self.train_params['xgb_model'])

            scores = model.attributes().get('best_msg').split('\t')[1:]
            metric_inds = np.where([self.overfit_metric in score for score in scores])[0]
            pt = float(scores[metric_inds[0]][(scores[metric_inds[0]].find(':') + 1):])
            pe = float(scores[metric_inds[1]][(scores[metric_inds[1]].find(':') + 1):])

            over_fit = (pt - pe) / ((pt + pe) / 2)
            if over_fit < self.train_params['max_over_fit']:
                self.best_score = self.base_score = {'train': pt, 'eval': pe}
                self.best_model = self.base_line_model = model
                self.xgb_params_best = self.xgb_params_start
                if self.verbose: print('Base model metrics :', pt, pe)
            else:
                if self.verbose: print('Over fit percent, currently no model without overfitting:  : ', over_fit, pt, pe)


        if self.__variable_not_set(self.xgb_params_best):
            temp_params = self.xgb_params_start
            curr_best_params = self.xgb_params_start
            curr_best_params = self.xgb_params_start
        else:
            temp_params = self.xgb_params_best
            curr_best_params = self.xgb_params_best

        # loop through the parameters to optimize(in order)
        start_time = time.time()
        for key in self.order:
            if (time.time() - start_time) / 60 > self.train_params['max_minutes_total']:
                print('fit_random has stopped due early due to the simulation reaching max minutes')
                break
            else:
                param_start_time = time.time()

                param_values = self.xgb_params[key]
                # this logic is due to the fact that when you assign the dictionary we are using for xgb_best_params
                # it actually assigns a pointer instead of the value, so we have to save the current best param on each
                # search and assign that to the xgb_best_params at the end of the search.
                if self.__variable_not_set(self.xgb_params_best) or self.__variable_not_set(self.xgb_params_best[key]):
                    best_param_value = self.xgb_params_start[key]
                else:
                    best_param_value =  self.xgb_params_best[key]

                temp_params = curr_best_params
                print('best_param: ',best_param_value)
                # Check if parameter has values to search on left and right( implies stored param vals as list )
                if self.__variable_not_set(param_values):
                    continue
                elif type(param_values[0]) != list: # coerce data to list type format if needed
                    param_values_list = [[vals for vals in param_values if vals!=best_param_value]] #exclude current best val for search
                else:
                    param_values_list = param_values
                # iterate through the left and right hand side values for each parameter
                for param_values in param_values_list:
                    # keeps track of how many iterations the search hasn't improve the objective function
                    npast = 0
                    # make param_values an iterable object
                    if type(param_values) in [int, float, str]:
                        param_values = [param_values]

                    if len(param_values)!=0:
                        print('Searching', key, 'over following values:', param_values)
                        if not self.__variable_not_set(self.best_model):
                            print('Current best model metrics:',self.best_model.attributes().get('best_msg').split('\t')[1:])
                    else:
                        print(key,'has no values to tune, continuing.')
                    # iterate through the values for each parameter
                    for v in param_values:
                        print('__Total time:',np.round((time.time() - start_time) / 60,2),'mins. Current Param time:',np.round((time.time() - param_start_time) / 60,2),'mins.__' )
                        temp_params[key] = v
                        if self.verbose:print('Tuning parameter:', key, 'value', v)
                        model = xgboost.train(params=temp_params,
                                              dtrain=dtrain,
                                              evals=evals,
                                              num_boost_round=self.train_params['num_boost_rounds'],
                                              early_stopping_rounds=self.train_params['early_stopping_rounds'],
                                              maximize=self.train_params['maximize'],
                                              verbose_eval=False,
                                              feval=self.feval,
                                              xgb_model=self.train_params['xgb_model'])

                        self.n_models_tried += 1
                        scores = model.attributes().get('best_msg').split('\t')[1:]
                        print(scores)
                        metric_inds = np.where([self.overfit_metric in score for score in scores])[0]
                        pt = float(scores[metric_inds[0]][(scores[metric_inds[0]].find(':') + 1):])
                        pe = float(scores[metric_inds[1]][(scores[metric_inds[1]].find(':') + 1):])
                        self.tune_results['train'].append(pt)
                        self.tune_results['eval'].append(pe)
                        self.tune_results['param'].append(key)
                        self.tune_results['val'].append(v)
                        self.tune_results['temp_params'].append(temp_params)
                        # print(self.xgb_params_best)
                        # print(curr_best_params)

                        print('value of pe:',pe)
                        print('value of curr best eval:',self.best_score['eval'])
                        # find whether model overfit/improved eval metric
                        over_fit = (pt - pe) / ((pt + pe) / 2)
                        ## if a model has still not been found that doesn't overfit
                        if self.__variable_not_set(self.best_model):
                            if over_fit < self.train_params['max_over_fit']:
                                print('hit best_model is none loop')
                                self.best_score = {'train': pt, 'eval': pe}
                                self.best_model = model
                                curr_best_params = temp_params
                                self.xgb_params_best = temp_params
                                if self.verbose: print('Base model metrics :', pt, pe)
                            else:
                                if self.verbose: print('Over fit percent, currently no model without overfitting: ', over_fit, pt, pe)
                            continue

                        ## if there is a current best model, determine if current iteration has better eval metric w/o overfitting
                        else:
                            if over_fit > self.train_params['max_over_fit']:
                                not_over_fit = False
                                if self.verbose:print('Over fit percent : ', over_fit, pt, pe)
                            else:
                                not_over_fit = True
                            if self.train_params['maximize'] and pe > self.best_score['eval']:
                                score_beaten = True
                            elif self.train_params['maximize'] == False and pe < self.best_score['eval']:
                                score_beaten = True
                            else:
                                score_beaten = False

                            if score_beaten and not_over_fit:
                                npast=0 #start search from newly found param value with reset build_past value
                                self.xgb_params_best[key] = best_param_value = v
                                self.best_score['train'] = pt
                                self.best_score['eval'] = pe
                                self.best_model = model
                                if self.verbose: print('**Score_beaten without overfitting**')
                                if self.verbose: print('*Base_score', self.base_score)
                                if self.verbose: print('*Best_score', self.best_score)
                            else:
                                if self.verbose: print('--Failed to beat score without overfitting. Train',pt,'Test',pe)
                                self.xgb_params_best[key] = best_param_value
                                npast += 1
                                if npast > self.train_params['build_past']:
                                    if self.verbose: print('Max build-past reached, continuing.')
                                    break
                        #break from the simulation if time expired
                        if (time.time() - start_time) / 60 > self.train_params['max_minutes_total']:
                            break
                        elif (time.time() - param_start_time) / 60 > self.train_params['max_mins_per_param']:
                            print('skipping to next param due to',key,'reaching max minutes')
                            break

                    if (time.time() - start_time) / 60 > self.train_params['max_minutes_total']:
                        break
        if self.__variable_not_set(self.best_model)==False:
            print('Tuning Finished. Resulting best model metrics:',self.best_model.attributes().get('best_msg').split('\t')[1:])
            self.train_results = self.best_model.eval(dtrain)
            self.eval_results = self.best_model.attributes().get('best_msg')
        else:
            print('Best model not found, consider increasing x.train_params["max_minutes"] or change overfit settings/params')

    def fit_random_search(self, dtrain, evals):
        '''
        This method uses a random range search approach to solve for the optimal parameters of a model.
        Random sampling is pulled from uniform distribution with LB and UB specified as a tuple by the user
        This method currently doesn't default to searching lambda,alpha,colsample, but user input can add these
        :param dtrain: xgboost D Matrix with the labels set for traning data
        :param evals: example  evals=[(train, 'Train'), (test, "Test")]) where train and test are xgb D matrices
        :return: returns a fitted object, with best model and scores stored
        '''
        import xgboost
        import time
        import numpy as np
        import random
        self.__set_feval_model_metric()

        # if the optimization is multiclass,
        if self.xgb_params.get('objective').__contains__('multi'):
            self.xgb_params_start['num_class']=len(np.unique(dtrain.get_label()))

        start_time = time.time()
        temp_params = self.xgb_params_start
        for iter in range(self.train_params['num_rand_samples']):
            print('__Total time:', np.round((time.time() - start_time) / 60, 2), 'mins.__')
            if (time.time() - start_time)/60 > self.train_params['max_minutes_total']:
                print('fit_random has stopped due early due to the simulation reaching max minutes')
                break
            else:
                # generate current iteration's parameter values (random sample from xgb_params range)
                for key in self.random_search_params:
                    # for each parameter in randsearch, pick a random value from the provided range in xgb_params
                    temp_params[key] =round(random.choice(self.xgb_params.get(key)),2)
                print([(key, value) for key, value in self.xgb_params_start.items() if key in self.random_search_params])

                model = xgboost.train(params=temp_params,
                                      dtrain=dtrain,
                                      evals=evals,
                                      num_boost_round=self.train_params['num_boost_rounds'],
                                      early_stopping_rounds=self.train_params['early_stopping_rounds'],
                                      maximize=self.train_params['maximize'],
                                      verbose_eval=False,
                                      feval=self.feval,
                                      xgb_model=self.train_params['xgb_model'])
                self.n_models_tried += 1
                scores = model.attributes().get('best_msg').split('\t')[1:]
                print(scores)
                metric_inds = np.where([self.overfit_metric in score for score in scores])[0]
                pt = float(scores[metric_inds[0]][(scores[metric_inds[0]].find(':') + 1):])
                pe = float(scores[metric_inds[1]][(scores[metric_inds[1]].find(':') + 1):])
                self.tune_results['train'].append(pt)
                self.tune_results['eval'].append(pe)
                self.tune_results['temp_params'].append(temp_params)


                over_fit = (pt - pe) / ((pt + pe) / 2)
                if bool(self.best_model) is False:
                    if over_fit < self.train_params['max_over_fit']:
                        self.best_score = {'train': pt, 'eval': pe}
                        self.best_model = model

                        for key in self.xgb_params_start.keys():
                            self.xgb_params_best[key] = self.xgb_params_start[key]
                        if self.verbose: print('Base model metrics :',pt,pe)
                    else:
                        if self.verbose: print('Over fit percent : ', over_fit, pt, pe)
                    continue

                # find whether model overfit/improved eval metric
                if over_fit > self.train_params['max_over_fit']:
                    not_over_fit = False
                    if self.verbose:print('Over fit percent : ', over_fit, pt, pe)
                else:
                    not_over_fit = True
                if self.train_params['maximize'] and pe > self.best_score['eval']:
                    score_beaten = True
                elif self.train_params['maximize'] == False and pe < self.best_score['eval']:
                    score_beaten = True
                else:
                    score_beaten = False

                if score_beaten and not_over_fit:
                    #this is once again due to the fact that assigning dictionary to a variable keeps a pointer to the original variable
                    # so in the next iteration, when we change the variable 'temp_params' the 'xgb_params_best' gets modified as well
                    for param in self.random_search_params:
                        self.xgb_params_best[param] = temp_params[param]
                    self.best_score['train'] = pt
                    self.best_score['eval'] = pe
                    self.best_model = model
                    if self.verbose: print('*Score_beaten without overfitting*')
                    if self.verbose: print('Base_score', self.base_score)
                    if self.verbose: print('Best_score', self.best_score)
                else:
                    if self.verbose: print('-Failed to beat score without overfitting. Train',pt,'Test',pe)
        if self.__variable_not_set(self.best_model)==False:
            print('Tuning Finished. Resulting best model metrics:',self.best_model.attributes().get('best_msg').split('\t')[1:])
            self.train_results = self.best_model.eval(dtrain)
            self.eval_results = self.best_model.attributes().get('best_msg')
        else:
            print('Best model not found, consider increasing x.train_params["max_minutes"] or change overfit settings/params')
