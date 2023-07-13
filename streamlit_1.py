import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.set_page_config(layout="wide")

def pred(st):
    bg_img = """
    <style>
        [data-testid='stAppViewContainer']{
            font-style: roboto;
            background-image: url(https://images4.alphacoders.com/206/206954.jpg);
            background-size: cover;
        }
        [data-testid='stSidebar']{
            background-image: url(https://wallpapers.com/images/high/colorful-abstract-art-on-samsung-full-hd-9zgi6iabu6uys5ne.webp);
            background-size: cover;
        }
        h1{
            color: #ffffff;
            text-align: Center;
        }
        h3{
            color: #ffffff;
        }

        .stMarkdown p {
            color: #ffffff;
        }

    </style>
    """
    st.markdown(bg_img, unsafe_allow_html=True)
    st.title('Prediction Penguins')
    penguin_file = st.file_uploader('Upload your own penguin data')
    if penguin_file is None:
        rf_pickle = open('random_forest_penguin.pickle', 'rb')
        map_pickle = open('output_penguin.pickle', 'rb')
        rfc = pickle.load(rf_pickle)
        unique_penguin_mapping = pickle.load(map_pickle)
        rf_pickle.close()
        map_pickle.close()
    else:
        penguin_df = pd.read_csv(penguin_file)
        penguin_df = penguin_df.dropna()
        output = penguin_df.species
        features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                               'flipper_length_mm', 'body_mass_g', 'sex']]
        features = pd.get_dummies(features)
        output, unique_penguin_mapping = pd.factorize(output)
        X_train, X_test, y_train, y_test = train_test_split(features, output, test_size=.8, random_state=42)
        rfc = RandomForestClassifier(random_state=42)
        rfc.fit(X_train, y_train)
        y_pred = rfc.predict(X_test)
        score = round(accuracy_score(y_pred, y_test), 2)
        st.write('we trained a random forest model on these data,'
                 'it has a score of {}! use the''input below to try out the model.'.format(score))
        with st.form('user_input'):
            island = st.selectbox('Penguin Island', options=[
            'Biscoe', 'Dream', 'Torgerson'])
            sex = st.selectbox('Sex', options=['Female', 'Male'])
            bill_length = st.number_input('Bill Length (mm)', min_value=0)
            bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
            flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
            body_mass = st.number_input('Body Mass (g)', min_value=0)
            st.form_submit_button()
        island_biscoe, island_dream, island_torgerson = 0, 0, 0
        sex_female, sex_male = 0, 0 
        if island == 'Biscoe':
            island_biscoe = 1
        elif island == 'Dream':
            island_dream = 1
        elif island == 'Torgerson':
            island_torgerson = 1
            sex_female, sex_male = 0, 0
        if sex == 'Female':
            sex_female = 1
        elif sex == 'Male':
            sex_male = 1
        new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length,\
                                    body_mass, island_biscoe, island_dream,\
                                    island_torgerson, sex_female, sex_male]])
        prediction_species = unique_penguin_mapping[new_prediction][0]
        st.write('We predict your penguin is of the {} species'.format(prediction_species))


def viz(st):
    
    bg_img = """
    <style>
        [data-testid='stAppViewContainer']{
            font-style: roboto;
            background-image: url(https://images4.alphacoders.com/206/206954.jpg);
            background-size: cover;
            
        }
        [data-testid='stSidebar']{
            background-image: url(https://wallpapers.com/images/high/colorful-abstract-art-on-samsung-full-hd-9zgi6iabu6uys5ne.webp);
            background-size: cover;
        }
    </style>
    """
    st.markdown(bg_img, unsafe_allow_html=True)
    st.title('Visualization By Seaborn')
    col1, col2 = st.columns(2)
    # selected_species = st.selectbox('what species would you like to visualize?',
    #                                 ['Adelie', 'Gentoo', 'Chinstrap'])
    with col1:
        selected_x_var = st.selectbox('what do you want the x variable to vbe?',
                                    ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'])
        selected_y_var = st.selectbox('what about the y?',
                                    ['bill_depth_mm', 'bill_length_mm', 'flipper_length_mm', 'body_mass_g'])
        penguins_df = pd.read_csv('penguins.csv')
        markers = {"Adelie": "X", "Gentoo": "s", "Chinstrap":'o'}
        fig, ax = plt.subplots()
        fig.patch.set_facecolor('none') 
        ax = sns.scatterplot(penguins_df, x=selected_x_var, y=selected_y_var,
                            hue='species', markers=markers, style='species' )
        plt.xlabel(selected_x_var, color='w')
        plt.ylabel(selected_y_var)
        fig.savefig('scaterplot.png', transparent=True)
        st.pyplot(fig)
    with col2:
        penguin_df = pd.read_csv('penguins.csv')
        penguin_df = penguin_df.dropna()
        output = penguin_df.species
        features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
                            'flipper_length_mm', 'body_mass_g', 'sex']]
        features = pd.get_dummies(features)
        output, uniques = pd.factorize(output)
        x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=.8)
        rfc = RandomForestClassifier(random_state=15)
        rfc.fit(x_train, y_train)
        y_pred = rfc.predict(x_test)
        score = accuracy_score(y_pred, y_test)
        print('Our accuracy score for this model is {}'.format(score))

        # Save the trained model
        with open('random_forest_penguin.pickle', 'wb') as rf_pickle:
            pickle.dump(rfc, rf_pickle)

        # Save the unique mapping
        with open('output_penguin.pickle', 'wb') as output_pickle:
            pickle.dump(uniques, output_pickle)

        # Plot feature importance
        fig, ax = plt.subplots(alpha=0.5)
        fig.patch.set_facecolor('none') 
        ax = sns.barplot(x=features.columns, y=rfc.feature_importances_, ax=ax, width=.7)
        plt.title('Feature Importance', color='w')
        plt.xticks(rotation=90, color='w')
        plt.yticks(color='w')
        plt.xlabel('Importance', color='w')
        plt.ylabel('Feature', color='w')
        plt.tight_layout()
        fig.savefig('feature_importance.png', transparent=True)

        # Perform prediction on new data
        input_bill_length_mm = 42.0  # Replace with actual input
        input_bill_depth_mm = 15.0  # Replace with actual input
        input_flipper_length_mm = 200.0  # Replace with actual input
        input_body_mass_g = 4000.0  # Replace with actual input
        input_island_biscoe = 1  # Replace with actual input
        input_island_dream = 0  # Replace with actual input
        input_island_torgerson = 0  # Replace with actual input
        input_sex_female = 1  # Replace with actual input
        input_sex_male = 0  # Replace with actual input

        new_prediction = rfc.predict([[input_bill_length_mm, input_bill_depth_mm, input_flipper_length_mm,
                                    input_body_mass_g, input_island_biscoe, input_island_dream,
                                    input_island_torgerson, input_sex_female, input_sex_male]])
        prediction_species = uniques[new_prediction][0]

        # Display the prediction
        st.image('feature_importance.png')


def main():
    bg_img_1 = """
    <style>
        [data-testid='stAppViewContainer']{
            background-image: url(https://4kwallpapers.com/images/walls/thumbs_3t/9771.jpg);
            background-size: cover;
        }
        [data-testid='stSidebar']{
            background-image: url(https://wallpapers.com/images/high/colorful-abstract-art-on-samsung-full-hd-9zgi6iabu6uys5ne.webp);
            background-size: cover;
        }

        [data-testid="stHeader"]{
            background-color: rgba(0, 0, 0, 0);
        }

        h1{
            color: #000000;
            text-align: Center;
        }
        h3{
            color: #000000;
        }

        .stMarkdown p {
            color: #000000;
        }

        .css-k3w14i{
            color: #000000;
        }
        </>
    """

    st.markdown(bg_img_1, unsafe_allow_html=True)
    st.sidebar.title('Sidebar')
    menu_item = ['Dataset', 'Visualization', 'Prediction']
    selected_item = st.sidebar.selectbox('Select Menu', menu_item)

    if selected_item == 'Dataset':
        st.title("Palmer's Penguins")
        penguins_df = pd.read_csv('penguins.csv')
        st.write(penguins_df.head(), width=700)
    elif selected_item == 'Prediction':
        pred(st)
    else :
        viz(st)


    

if __name__ == '__main__':
    main()
