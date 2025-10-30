"""
Módulo de explicabilidad usando SHAP para el modelo de valuación de propiedades.
"""

import numpy as np
import pandas as pd
import shap
import joblib
from typing import Dict, List, Any, Optional


class SHAPExplainer:
    """Clase para generar explicaciones SHAP del modelo de valuación."""

    def __init__(self, model_path: str = 'modelo_valoracion.joblib'):
        """
        Inicializa el explicador SHAP.

        Args:
            model_path: Ruta al archivo del modelo serializado
        """
        self.pipeline = joblib.load(model_path)

        # Si es un Pipeline de sklearn, extraer el modelo final
        if hasattr(self.pipeline, 'named_steps'):
            self.model = self.pipeline.named_steps['model']
            self.preprocessor = self.pipeline.named_steps.get('preprocessor', None)
        else:
            # Si no es pipeline, asumir que es el modelo directo
            self.model = self.pipeline
            self.preprocessor = None

        # Obtener nombres de features del pipeline
        if hasattr(self.pipeline, 'feature_names_in_'):
            self.feature_columns = list(self.pipeline.feature_names_in_)
        else:
            self.feature_columns = []

        # Inicializar el explicador SHAP con TreeExplainer (para Random Forest)
        try:
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_available = True
            print("SHAP TreeExplainer inicializado correctamente")
        except Exception as e:
            print(f"Error al inicializar SHAP: {e}")
            self.explainer = None
            self.shap_available = False

    def _translate_feature_name(self, feature: str) -> str:
        """
        Traduce nombres técnicos de features a nombres legibles en español.

        Args:
            feature: Nombre técnico de la feature

        Returns:
            Nombre legible en español
        """
        translations = {
            'superficie_total': 'Superficie total',
            'superficie_util': 'Superficie útil',
            'dormitorios': 'Dormitorios',
            'banos': 'Baños',
            'latitud': 'Latitud',
            'longitud': 'Longitud',
            'es_santiago': 'Es Santiago',
            'superficie_ratio': 'Ratio superficie útil/total',
            'hab_por_dormitorio': 'M² por dormitorio',
            'banos_dormitorios_ratio': 'Ratio baños/dormitorios',
            'icvu_promedio': 'ICVU promedio',
            'tipo_encoded': 'Tipo de propiedad',
            'region_encoded': 'Región',
            'superficie_per_room': 'M² por habitación',
            'is_premium': 'Segmento premium',
            'is_budget': 'Segmento económico',
            'distancia_ed_superior_km': 'Distancia educación superior',
            'distancia_ed_escolar_km': 'Distancia educación escolar',
            'distancia_comisaria_km': 'Distancia comisaría',
            'distancia_est_salud_km': 'Distancia centro de salud',
            'distancia_metro_km': 'Distancia metro'
        }
        return translations.get(feature, feature)

    def explain_prediction(
        self,
        X: pd.DataFrame,
        top_n: int = 5
    ) -> Optional[Dict[str, Any]]:
        """
        Genera explicación SHAP para una predicción.

        Args:
            X: DataFrame con las features de la propiedad
            top_n: Número de features más importantes a retornar

        Returns:
            Diccionario con la explicación SHAP o None si no está disponible
        """
        if not self.shap_available:
            return None

        try:
            # Si hay un preprocessor, aplicarlo a los datos
            if self.preprocessor is not None:
                X_processed = self.preprocessor.transform(X)

                # Obtener nombres de features después del preprocesamiento
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    processed_feature_names = list(self.preprocessor.get_feature_names_out())
                else:
                    processed_feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

                # Convertir a DataFrame si es necesario
                if not isinstance(X_processed, pd.DataFrame):
                    X_processed = pd.DataFrame(X_processed, columns=processed_feature_names)
            else:
                X_processed = X
                processed_feature_names = list(X.columns)

            # Calcular valores SHAP
            shap_values = self.explainer.shap_values(X_processed)

            # Si shap_values es una lista (multi-output), tomar el primer elemento
            if isinstance(shap_values, list):
                shap_values = shap_values[0]

            # Obtener valores para la primera predicción
            if len(shap_values.shape) > 1:
                shap_values_instance = shap_values[0]
            else:
                shap_values_instance = shap_values

            # Obtener el valor base (expected value)
            if isinstance(self.explainer.expected_value, (list, np.ndarray)):
                base_value = self.explainer.expected_value[0]
            else:
                base_value = self.explainer.expected_value

            # Crear lista de features con sus valores SHAP
            feature_impacts = []
            for i, feature in enumerate(processed_feature_names):
                if i < len(shap_values_instance):
                    feature_impacts.append({
                        'feature': feature,
                        'feature_raw': feature,
                        'shap_value': float(shap_values_instance[i]),
                        'feature_value': float(X_processed.iloc[0, i]) if isinstance(X_processed, pd.DataFrame) else float(X_processed[0, i])
                    })

            # Ordenar por valor absoluto de SHAP (impacto)
            feature_impacts.sort(key=lambda x: abs(x['shap_value']), reverse=True)

            # Tomar top N features
            top_features = feature_impacts[:top_n]

            # Generar texto explicativo
            explanation_text = self._generate_explanation_text(
                base_value,
                top_features
            )

            return {
                'available': True,
                'base_value': float(base_value),
                'top_features': [
                    {
                        'feature': self._translate_feature_name(f['feature']),
                        'feature_raw': f['feature_raw'],
                        'shap_value': f['shap_value'],
                        'feature_value': f['feature_value'],
                        'impact_direction': 'increase' if f['shap_value'] > 0 else 'decrease'
                    }
                    for f in top_features
                ],
                'explanation_text': explanation_text
            }

        except Exception as e:
            print(f"Error al calcular SHAP: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _generate_explanation_text(
        self,
        base_value: float,
        top_features: List[Dict[str, Any]]
    ) -> str:
        """
        Genera texto explicativo legible de la predicción.

        Args:
            base_value: Valor base del modelo
            top_features: Lista de features más importantes

        Returns:
            Texto explicativo formateado
        """
        # El base_value ya está en UF (el modelo predice directamente en UF)
        base_value_uf = base_value

        text = f"Explicación de la predicción (valor base: {base_value_uf:.2f} UF):\n\n"
        text += "Factores que más influyen en el precio:\n"

        for i, feature in enumerate(top_features, 1):
            feature_name = self._translate_feature_name(feature['feature'])
            shap_value = feature['shap_value']
            feature_value = feature['feature_value']

            # El SHAP value ya está en UF (contribución directa al precio)
            impact_uf = shap_value

            direction = "incrementa" if shap_value > 0 else "reduce"
            sign = "+" if shap_value > 0 else ""

            text += f"{i}. {feature_name} (valor: {feature_value:.2f}): "
            text += f"{direction} el precio en ~{sign}{impact_uf:.2f} UF "
            text += f"(SHAP: {sign}{shap_value:.4f})\n"

        return text


def get_shap_explanation_for_prediction(
    model_path: str,
    X: pd.DataFrame,
    top_n: int = 5
) -> Optional[Dict[str, Any]]:
    """
    Función de utilidad para obtener explicación SHAP de una predicción.

    Args:
        model_path: Ruta al modelo serializado
        X: DataFrame con las features
        top_n: Número de features principales a retornar

    Returns:
        Diccionario con explicación SHAP o None
    """
    try:
        explainer = SHAPExplainer(model_path)
        return explainer.explain_prediction(X, top_n)
    except Exception as e:
        print(f"Error al generar explicación SHAP: {e}")
        return None
