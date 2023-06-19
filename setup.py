from setuptools import find_packages, setup


param_to_drop = '-e .'
def get_requirements(file_path:str):
    ''''
    This function will return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [requirement.replace('\n', '') for requirement in requirements]
    
        if param_to_drop in requirements:
            requirements.remove(param_to_drop)
    
    return requirements
        

setup(
    name='Students Performance Project',
    version='0.0.1',
    author='Luis Munoz',
    author_email='luisezequielmunoz@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)