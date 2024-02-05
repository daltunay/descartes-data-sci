import streamlit as st
from classification.processing import DataProcessor
from classification.model import ModelPredictor

st.set_page_config(
    page_title="Descartes Underwriting - ALTUNAY Daniel",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="auto",
)

st.caption(
    """
    This web app is my take for the _Descartes Underwriting - Data Scientist Technical Test_.  
    It was built and deployed with [Streamlit](https://streamlit.io), and primarily uses the following libraries: `pandas`, `sklearn`, `lazypredict`, `matplotlib`.

    If you have any questions, feel free to contact me: [daniel.altunay@gmail.com](mailto:daniel.altunay@gmail.com)
    """
)

st.divider()

st.header("Input Data")

st.markdown(
    """
| Column name | Definition | Unit |
|--------|-----------|--------|
| `ignition`   | Target column| Boolean: {1,0} |
| `distance_{feature}` | Distance to nearest feature  |  Meters (m) |
| vegetation class: from `cropland` to `wetland`  |  Ratio of each of the vegetation classes  under which the ignition or non-ignition point lies    |  No unit (between 0 and 1)  |
| `aspect`  |  Orientation of the slope    |  Degrees (°)  |
| `elevation`  |  elevation value    |  Meters  |
| `slope`  |  Slope value    |  Degrees (°)  |
| `pop_dens`  |  Population density value    |  Persons per km2  |
| `max_temp`  |  Maximum temperature of the day    |  Degrees celsius (°C)  |
| `avg_temp`  |  Average temperature of the day   |  Degrees celsius (°C)  |
| `max_wind_vel`  |  Maximum wind velocity of the day    |  Meters per second (m/s)  |
| `avg_wind_angle`  |  Average angle of the vector wind over the day    |  Degrees (°)  |
| `avg_rel_hum`  |  Average relative humidity over the day    |  %  |
| `avg_soil`  |  Average soil moisture of the day    |  m3/m3  |
| `sum_prec`  |  Cumulative rainfall precipitation of the day    |  Millimeters (mm)  |
| `yearly_avg_temp`  |  Average temperature over the year    |  Degrees celsius (°C)  |
| `anom_{feature}`  |  Standardized anomaly of weather for the given day over the last 30 years. When the anomaly is positive, it means that the feature value is greater that the 30-year average    |  No unity |
| `forest`  |  Sum of all the columns where the names start with `forest`   |  No unit  |
| `vegetation_class`  |  Vegetation with the max occurrence in the vicinity of the ignition/non-ignition point    |  Without unit  |
| `Year`  |  Year of ignition    |  Without unit  |
| `max_max_temp`  |  Missing information    |  Missing information  |
"""
)

st.header("Preprocessing")

processor = DataProcessor()

cols = st.columns(3)
random_state = cols[0].number_input("Random state", value=0, step=1)
test_size = cols[1].slider(
    "Test size", min_value=0.1, max_value=0.9, value=0.25, step=0.05
)
shuffle = cols[2].checkbox("Shuffle", value=True)
stratify = cols[2].checkbox("Stratify", value=True)

X_train, X_test, y_train, y_test = processor.split_data(
    test_size=test_size,
    random_state=random_state,
    shuffle=shuffle,
    stratify_flag=stratify,
)

st.header("Visualizing Data")
with st.expander("Click to expand", expanded=False):
    cols = st.columns(4)
    X_container = st.container()
    with X_container:
        st.subheader("`X` data")
        cols = st.columns(2)
        cols[1].dataframe(X_train, use_container_width=True)
        cols[0].dataframe(X_train.describe(), use_container_width=True)
    y_container = st.container()
    with y_container:
        st.subheader("`y` data")
        cols = st.columns(2)
        cols[1].dataframe(y_train, use_container_width=True)
        cols[0].dataframe(y_train.describe(), use_container_width=True)

st.header("Classification")
st.markdown(
    """
    Here, I am using the [`lazypredict`](https://github.com/shankarpandala/lazypredict) package to perform classification.
    > _Lazy Predict helps build a lot of basic models without much code and helps understand which models works better without any parameter tuning._
    """
)
if (
    st.columns([2, 1, 2])[1].button(
        "TRAIN CLASSIFIERS",
        type="primary",
        use_container_width=True,
    )
    or "predictor" in st.session_state
):
    predictor = ModelPredictor(random_state)
    st.session_state["predictor"] = predictor
    models, predictions = predictor.lazy_predict(X_train, X_test, y_train, y_test)

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Model performance:")
        st.dataframe(models)
    with cols[1]:
        st.subheader("Model predictions:")
        st.dataframe(predictions)

    st.divider()
    model_dict = predictor.provide_models(X_train, X_test, y_train, y_test)
    st.write("Please select a model to see its specific results:")
    if selected_model := model_dict.get(
        st.selectbox("Select model", model_dict.keys(), index=None)
    ):

        st.header("Model Results")
        with st.columns(3)[1]:
            st.subheader("Classification Report")
            try:
                st.dataframe(
                    predictor.get_classification_report(X_test, y_test, selected_model),
                    use_container_width=True,
                )
            except Exception as e:
                st.warning("Not availaible for this model")
        cols = st.columns(2, gap="large")
        with cols[0]:
            st.subheader("Confusion Matrix")
            try:
                with st.spinner():
                    st.pyplot(
                        predictor.get_confusion_matrix(X_test, y_test, selected_model)
                    )
            except:
                st.warning("Not availaible for this model")
        with cols[1]:
            st.subheader("ROC Curve")
            try:
                with st.spinner():
                    st.pyplot(predictor.get_roc_curve(X_test, y_test, selected_model))
            except Exception as e:
                st.warning("Not availaible for this model")
        cols = st.columns(2, gap="large")
        with cols[0]:
            st.subheader("Precision Recall Curve")
            try:
                with st.spinner():
                    st.pyplot(
                        predictor.get_precision_recall_curve(
                            X_test, y_test, selected_model
                        )
                    )
            except Exception as e:
                st.warning("Not availaible for this model")
        with cols[1]:
            st.subheader("Feature Importance")
            try:
                with st.spinner():
                    st.pyplot(
                        predictor.get_feature_importance(
                            selected_model, feature_names=X_test.columns, top_n=15
                        )
                    )
            except Exception as e:
                st.warning("Not availaible for this model")
