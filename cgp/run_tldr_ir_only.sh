# 26 July 2024

jbsub -q platform -name const-llama3-70-tldr-ir-only -mem 128G -cores 1x2+2 -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-const-llama3-70-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-const-llama3-70-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py /dccstor/ai4code-summ/benchmark-paper/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c/ /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/llama3-70-colbert-top-1-const-ir-only.parquet 1

# jbsub -name const-llama3-70-tldr-ir-only -mem 128G -cores 1x2+2 -require a100_40gb -err /dccstor/rhassistant/platform_training/logs/const-const-llama3-70-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-const-llama3-70-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py /dccstor/ai4code-summ/benchmark-paper/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/7129260dd854a80eb10ace5f61c20324b472b31c/ /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/llama3-70-colbert-top-1-const-ir-only.parquet 1

# jbsub -q platform -name const-starcoder2-15b-instruct-v0.1-tldr-ir-only -mem 128G -cores 1x2+2  -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-starcoder2-15b-instruct-v0.1-1-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-starcoder2-15b-instruct-v0.1-1-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py /dccstor/rhassistant/platform_training/models/starcoder2-15b-instruct-v0.1 /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/starcoder2-15b-instruct-v0.1-colbert-top-1-const-ir-only.parquet 1


# jbsub -q platform -name const-codellama-34b-colbert-1-tldr-ir-only -mem 128G -cores 1x2+2  -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-codellama-34b-colbert-1-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-codellama-34b-colbert-1-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py codellama/CodeLlama-34b-Instruct-hf /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/codellama-34b-colbert-top-1-const-ir-only.parquet 1

# jbsub -q platform -name const-codellama-34b-colbert-3-tldr-ir-only -mem 128G -cores 1x2+2  -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-codellama-34b-colbert-3-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-codellama-34b-colbert-3-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py codellama/CodeLlama-34b-Instruct-hf /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/codellama-34b-colbert-top-3-const-ir-only.parquet 3

# jbsub -q platform -name const-codellama-34b-colbert-10-tldr-ir-only -mem 128G -cores 1x2+2  -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-codellama-34b-colbert-10-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-codellama-34b-colbert-10-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py codellama/CodeLlama-34b-Instruct-hf /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/codellama-34b-colbert-top-10-const-ir-only.parquet 10


# jbsub -q platform -name const-codellama-7b-colbert-1-tldr-ir-only -mem 128G -cores 1x1+1  -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-codellama-7b-colbert-1-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-codellama-7b-colbert-1-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py /dccstor/rhassistant/platform_training/training/codellama-7b/nl2yaml/ft_tldr/epochs/epoch-2.0/ /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/codellama-7b-ft-colbert-top-1-const-ir-only.parquet 1

# jbsub -q platform -name const-codellama-7b-colbert-3-tldr-ir-only -mem 128G -cores 1x1+1  -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-codellama-7b-colbert-3-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-codellama-7b-colbert-3-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py /dccstor/rhassistant/platform_training/training/codellama-7b/nl2yaml/ft_tldr/epochs/epoch-2.0/ /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/codellama-7b-ft-colbert-top-3-const-ir-only.parquet 3

# jbsub -q platform -name const-codellama-7b-colbert-10-tldr-ir-only -mem 128G -cores 1x1+1  -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-codellama-7b-colbert-10-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-codellama-7b-colbert-10-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py /dccstor/rhassistant/platform_training/training/codellama-7b/nl2yaml/ft_tldr/epochs/epoch-2.0/ /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/codellama-7b-ft-colbert-top-10-const-ir-only.parquet 10


# jbsub -q platform -name const-codellama-7b-colbert-base-1-tldr-ir-only -mem 128G -cores 1x1+1  -require a100_80gb -err /dccstor/rhassistant/platform_training/logs/const-codellama-7b-colbert-base-1-tldr-ir-only.err -out /dccstor/rhassistant/platform_training/logs/const-codellama-7b-colbert-base-1-tldr-ir-only.out python /dccstor/rhassistant/mehant/paper/constrain_gen_pipeline_codellama/guidance_pipeline/gen_ir_only.py /dccstor/rhassistant/platform_training/models/codellama/CodeLlama-7b-hf /dccstor/rhassistant/mehant/paper/tldr_type_split/tldr_split_dataset_test_ir_inf_clean.parquet /dccstor/rhassistant/mehant/paper/tldr_type_split/codellama-7b-base-colbert-top-1-const-ir-only.parquet 1


