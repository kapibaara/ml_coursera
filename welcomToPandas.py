import pandas

data = pandas.read_csv('data_seta/titanic.csv', index_col='PassengerId')

print(data['Sex'].value_counts())

count_of_survived = data['Survived'].where(data['Survived'] == 1).count()
survived = 342 / 891

print('процент выживших', "{:.2%}".format(survived))

count_of_first_class = data['Pclass'].where(data['Pclass'] == 1).count()
#
print('процент 1 класса', "{:.2}".format(0.414838))

print('корреляция пирсона\n', data[['SibSp', 'Parch']].corr(method="pearson"))

female_names = data.loc[data['Sex'] == 'female', 'Name']

def find_name(name):
    if 'Mrs.' in name:
        first_indx = name.find('(')
        last_indx = name.find(')')

        self_name = name[first_indx + 1:last_indx]

        return self_name.split()[0]
    else:
        return name.split()[2]

print(female_names.map(find_name).describe())

