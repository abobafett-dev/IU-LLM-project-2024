import torch
import transformers
from peft import LoraConfig, TaskType, PeftModel
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

        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.1,
            r=8,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("IlyaGusev/saiga_llama3_8b")
        model = transformers.AutoModelForCausalLM.from_pretrained(
            "IlyaGusev/saiga_llama3_8b",
            device_map={'': 0},
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config
        )
        model = PeftModel.from_pretrained(model, model_id="../saiga_lora2_ru", config=peft_config)
        self.model = model.merge_and_unload()

    def predict_tags(self, description: str, title: str, subtitle: str = '') -> dict:
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
        input_text = f"Заголовок: {title}"
        if subtitle != '':
            input_text += f"\nПодзаголовок: {subtitle}"
        input_text += f"\nОписание: {description}"
        formatted_prompt = self.tokenizer.apply_chat_template([{
            "role": "system",
            "content": system_prompt
        }, {
            "role": "user",
            "content": input_text
        }], tokenize=False, add_generation_prompt=True)

        model_inputs = self.tokenizer([formatted_prompt], return_tensors="pt").to('cuda')

        generated_ids = self.model.generate(
            input_ids=model_inputs.input_ids,
            max_new_tokens=32,
            eos_token_id=self.tokenizer.encode('<|eot_id|>')[0],
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        response = response.splitlines()[0].split(", ")
        result = []
        stop_suffix = 'assist'
        for string in response:
            if string.endswith(stop_suffix):
                result.append(string[:-len(stop_suffix)])
                break
            else:
                result.append(string)

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
