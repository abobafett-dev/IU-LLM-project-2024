import torch
import transformers
from peft import PeftModel
import json


class PipelineTags:
    def __init__(self, load_in_4_bit: bool = False, load_in_8_bit: bool = False):
        if sum([load_in_4_bit, load_in_8_bit]) > 1:
            raise ValueError("Only one of load_in_4_bit, load_in_8_bit can be True")
        elif sum([load_in_4_bit, load_in_8_bit]) == 0:
            quantization_config = None
        else:
            quantization_config = {
                "load_in_4bit": load_in_4_bit,
                "bnb_4bit_compute_dtype": torch.bfloat16,
                "load_in_8bit": load_in_8_bit
            }

        with open("../tags_ru.json", "r", encoding="UTF-8") as f:
            self.list_of_tags = json.load(f)

        tokenizer = transformers.AutoTokenizer.from_pretrained("IlyaGusev/saiga_llama3_8b")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "IlyaGusev/saiga_llama3_8b",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(model, model_id="XaPoHbomj/saiga_results_ru")
        model.eval()

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2**12
        )

    def predict_tags(self, input_prompt: str) -> dict:
        system_prompt = """Задача: Как ведущий редактор компании мирового уровня, вам поручено выполнить ключевую задачу: провести детальный анализ предоставленного текста и выделить ключевые слова и фразы, которые наиболее точно отражают его содержание. Эти ключевые слова и фразы будут использованы для создания тегов, которые должны быть релевантными, конкретными и краткими. Теги помогут пользователям быстро понять основные темы и характеристики текста.

        Контекст и мотивация: Успешное выполнение этой задачи имеет решающее значение для будущего компании, в которой вы работаете уже 20 лет. От вашей работы зависит не только имидж компании, но и ваша собственная карьера, включая премии и потенциальную долю в компании. Поэтому крайне важно подойти к задаче с максимальной тщательностью и профессионализмом.
        
        Инструкция по выполнению задачи:
        
        Анализ текста: Внимательно прочитайте текст, чтобы понять его основной смысл, контекст и основные идеи.
        
        Выделение ключевых слов и фраз: Определите наиболее важные слова и фразы, которые наиболее точно отражают содержание текста. Обратите внимание на уникальные термины и фразы, которые чётко описывают темы и характеристики.
        
        Создание тегов: На основе выделенных ключевых слов и фраз сформулируйте теги. Теги должны быть:
        
        Релевантными: Полностью отражать содержание текста.
        
        Конкретными: Учитывать специфические аспекты, а не общие термины.
        Краткими: Один-два слова для обеспечения лёгкости восприятия.
        
        Примечания:
        Старайтесь избегать общих слов и фраз. Выбирайте наиболее точные термины.
        Учитывайте, что теги должны быть понятны широкой аудитории и точно отражать основные идеи текста.
        Текст для анализа:
        """

        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": input_prompt
            }
        ]

        response = self.pipeline(
            messages,
            pad_token_id=self.pipeline.tokenizer.eos_token_id
        )[0]['generated_text'][-1]['content']

        result = response.split(", ")

        tags = {
            "Тип данных": [],
            "Предметная область": [],
            "География и достопримечательности": [],
            "Техники": [],
            "Задача": [],
            "Язык": []
        }

        for key, values in self.list_of_tags.items():
            filtered_values = [value for value in values if value in result]
            if filtered_values:
                tags[key] = filtered_values

        return tags
