from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv


load_dotenv()


def generate_pet_name(animal_type, pet_color):

    #temperature how creative the model will be 
    llm = OpenAI(temperature=0.8)


    #prompt templates
    prompt_template_name = PromptTemplate(
        input_variables=['animal_type', 'pet_color'] ,
        template="I have a {animal_type} pet and I want a cool name for it, it is {pet_color} in color. Suggest me five cool names for my pet."
    )


    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key='pet_names')

    response = name_chain({'animal_type': animal_type, 'pet_color': pet_color})

    #responses
    # {'animal_type': 'horse', 'pet_color': 'black', 'text': '\n\n1. Midnight\n2. Shadow\n3. Raven\n4. Eclipse\n5. Coal'}
    # {'animal_type': 'horse', 'pet_color': 'black', 'pet_names': '\n\n1. Shadow\n2. Midnight\n3. Jet\n4. Raven\n5. Onyx'}

    return response 


# if __name__ == "__main__" :
#     print(generate_pet_name("horse","black"))


