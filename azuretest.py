from azureml.core.workspace import Workspace
from azureml.core.runconfig import RunConfiguration
from azureml.core.image import ContainerImage
from azureml.core import ScriptRunConfig, Experiment
from azureml.core.model import Model
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.webservice import AciWebservice, Webservice


# ws = Workspace.create(name='demoGarbarino',
#                       subscription_id='2dbdb5d8-da03-41b7-b36d-eb5574408ca6',
#                       resource_group='demoGarbarino',
#                       create_resource_group=False,
#                       location='eastus2')

ws = Workspace.from_config('config.json')
exp = Experiment(workspace=ws, name='garbarinoExperiment')

run_local = RunConfiguration()
run_local.environment.python.user_managed_dependencies = True

# src = ScriptRunConfig(source_directory='', script='train_local.py', run_config=run_local)

# run = exp.submit(src)
# run.wait_for_completion(show_output=True)

model = Model.register(model_path="model",
                       model_name="model",
                       tags={'type': "RNN", 'version': 55},
                       description="LSTM model to predict sales",
                       workspace=ws)


env = CondaDependencies.create(conda_packages=['keras', 'pandas', 'scikit-learn'])
env.save_to_file(base_directory='./', conda_file_path='env.yml')

aciconfig = AciWebservice.deploy_configuration(cpu_cores=1,
                                               auth_enabled=True,
                                               memory_gb=1,
                                               tags= {'name':'LSTM', 'framework':'Keras'},
                                               description='LSTM model to predict sales')

image_config = ContainerImage.image_configuration(runtime= "python",
                                                  execution_script="script.py",
                                                  conda_file="env.yml")

service = Webservice.deploy_from_model(workspace=ws,
                                       name='keras-lstm',
                                       deployment_config= aciconfig,
                                       models=[model],
                                       image_config=image_config)

service.wait_for_deployment(show_output=True)
print(service.get_keys())
print(service.get_logs())


