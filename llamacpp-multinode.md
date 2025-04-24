## Single node

Run:

```
llama-cli -hf unsloth/gemma-3-4b-it-GGUF:Q8_0 -cnv
```

## Multinode

On node =nid002998=
`rpc-server --host 0.0.0.0 -p 6666 -c`

On node =nid002996=
`rpc-server --host 0.0.0.0 -p 6666 -c`

On client node
`llama-cli -hf unsloth/gemma-3-4b-it-GGUF:Q8_0 -cnv --rpc "nid002998:6666,nid002996:6666" -cnv`

Ideally servers should be started through a SLURM batch job and the client from
interactive session.

## Serve API

Both client and server running on same node.

`llama-server -hf unsloth/gemma-3-4b-it-GGUF:Q8_0 --port 8080 --rpc "nid002998:6666,nid002996:6666" &`

Send request:
```
curl --request POST \
    --url http://localhost:8080/completion \
    --header "Content-Type: application/json" \
    --data '{"prompt": "who are you?"} | json_pp
```

=json_pp= is a tool to pretty print the JSON output.

The =content= field of the returned output contains the
output.

Output:

```
{
   "model" : "gpt-3.5-turbo",
   "stop_type" : "eos",
   "generation_settings" : {
      "dry_multiplier" : 0,
      "grammar_lazy" : false,
      "dynatemp_exponent" : 1,
      "grammar_triggers" : [],
      "speculative.n_min" : 0,
      "timings_per_token" : false,
      "temperature" : 0.800000011920929,
      "samplers" : [
         "penalties",
         "dry",
         "top_k",
         "typ_p",
         "top_p",
         "min_p",
         "xtc",
         "temperature"
      ],
      "presence_penalty" : 0,
      "n_keep" : 0,
      "top_p" : 0.949999988079071,
      "min_p" : 0.0500000007450581,
      "lora" : [],
      "mirostat" : 0,
      "chat_format" : "Content-only",
      "mirostat_tau" : 5,
      "n_probs" : 0,
      "n_discard" : 0,
      "frequency_penalty" : 0,
      "n_predict" : -1,
      "preserved_tokens" : [],
      "typical_p" : 1,
      "seed" : 4294967295,
      "dry_base" : 1.75,
      "stop" : [],
      "mirostat_eta" : 0.100000001490116,
      "repeat_last_n" : 64,
      "speculative.p_min" : 0.75,
      "dynatemp_range" : 0,
      "max_tokens" : -1,
      "repeat_penalty" : 1,
      "top_k" : 40,
      "logit_bias" : [],
      "post_sampling_probs" : false,
      "dry_sequence_breakers" : [
         "\n",
         ":",
         "\"",
         "*"
      ],
      "ignore_eos" : false,
      "dry_allowed_length" : 2,
      "xtc_probability" : 0,
      "grammar" : "",
       "min_keep" : 0,
      "xtc_threshold" : 0.100000001490116,
      "speculative.n_max" : 16,
      "dry_penalty_last_n" : 4096,
      "stream" : false
   },
   "tokens" : [],
   "stop" : true,
   "prompt" : "<bos>who are you?",
   "stopping_word" : "",
   "timings" : {
      "predicted_per_token_ms" : 45.3058823529412,
      "prompt_per_token_ms" : 54.906,
      "predicted_per_second" : 22.0721890418073,
      "prompt_ms" : 54.906,
      "prompt_n" : 1,
      "predicted_ms" : 2310.6,
      "predicted_n" : 51,
      "prompt_per_second" : 18.2129457618475
   },
   "has_new_line" : true,
   "id_slot" : 0,
   "tokens_cached" : 55,
   "tokens_predicted" : 51,
   "content" : "\n\nI am Gemma, an open-weights AI assistant. I’m a large language model created by the Gemma team at Google DeepMind. I am widely available to the public. I take text and images as inputs and output text only.\n",
   "index" : 0,
   "tokens_evaluated" : 5,
   "truncated" : false
}
```
