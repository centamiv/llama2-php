<?php

declare(strict_types=1);

ini_set('memory_limit', '-1');

class Config
{
    public int $dim;
    public int $hiddenDim;
    public int $nLayers;
    public int $nHeads;
    public int $nKvHeads;
    public int $vocabSize;
    public int $seqLen;

    public function __construct($fileHandle)
    {
        $data = unpack("i7", fread($fileHandle, 28));
        $this->dim = $data[1];
        $this->hiddenDim = $data[2];
        $this->nLayers = $data[3];
        $this->nHeads = $data[4];
        $this->nKvHeads = $data[5];
        $this->vocabSize = $data[6];
        $this->seqLen = $data[7];
    }
}

class TransformerWeights
{
    public SplFixedArray $tokenEmbeddingTable;
    public SplFixedArray $rmsAttWeight;
    public SplFixedArray $rmsFfnWeight;
    public SplFixedArray $wq;
    public SplFixedArray $wk;
    public SplFixedArray $wv;
    public SplFixedArray $wo;
    public SplFixedArray $w1;
    public SplFixedArray $w2;
    public SplFixedArray $w3;
    public SplFixedArray $rmsFinalWeight;
    public SplFixedArray $freqCisReal;
    public SplFixedArray $freqCisImag;
    public ?SplFixedArray $wcls = null;

    public function __construct(Config $config, $fileHandle)
    {
        $sharedWeights = $config->vocabSize > 0;
        if ($config->vocabSize < 0) {
            $config->vocabSize = -$config->vocabSize;
        }

        $headSize = (int)($config->dim / $config->nHeads);
        $kvDim = (int)($config->dim * $config->nKvHeads / $config->nHeads);

        $this->tokenEmbeddingTable = $this->readFloats($fileHandle, $config->vocabSize * $config->dim);
        $this->rmsAttWeight = $this->readFloats($fileHandle, $config->nLayers * $config->dim);
        $this->wq = $this->readFloats($fileHandle, $config->nLayers * $config->dim * $config->dim);
        $this->wk = $this->readFloats($fileHandle, $config->nLayers * $config->dim * $kvDim);
        $this->wv = $this->readFloats($fileHandle, $config->nLayers * $config->dim * $kvDim);
        $this->wo = $this->readFloats($fileHandle, $config->nLayers * $config->dim * $config->dim);
        $this->rmsFfnWeight = $this->readFloats($fileHandle, $config->nLayers * $config->dim);
        $this->w1 = $this->readFloats($fileHandle, $config->nLayers * $config->dim * $config->hiddenDim);
        $this->w2 = $this->readFloats($fileHandle, $config->nLayers * $config->hiddenDim * $config->dim);
        $this->w3 = $this->readFloats($fileHandle, $config->nLayers * $config->dim * $config->hiddenDim);
        $this->rmsFinalWeight = $this->readFloats($fileHandle, $config->dim);

        // Skip freq_cis_real and freq_cis_imag (computed on the fly)
        fseek($fileHandle, $config->seqLen * $headSize, SEEK_CUR);

        if (!$sharedWeights) {
            $this->wcls = $this->readFloats($fileHandle, $config->vocabSize * $config->dim);
        } else {
            $this->wcls = $this->tokenEmbeddingTable;
        }
    }

    private function readFloats($fileHandle, int $count): SplFixedArray
    {
        $floatsArray = new SplFixedArray($count);
        // Reading in chunks to avoid memory spikes and for speed
        $chunkSize = 10000;
        for ($offset = 0; $offset < $count; $offset += $chunkSize) {
            $currentChunkSize = min($chunkSize, $count - $offset);
            $binaryData = fread($fileHandle, $currentChunkSize * 4);
            $unpackedFloats = unpack("f*", $binaryData);
            for ($index = 0; $index < $currentChunkSize; $index++) {
                $floatsArray[$offset + $index] = $unpackedFloats[$index + 1];
            }
        }
        return $floatsArray;
    }
}

class RunState
{
    public SplFixedArray $x;
    public SplFixedArray $xb;
    public SplFixedArray $xb2;
    public SplFixedArray $hb;
    public SplFixedArray $hb2;
    public SplFixedArray $q;
    public SplFixedArray $k;
    public SplFixedArray $v;
    public SplFixedArray $att;
    public SplFixedArray $logits;
    public SplFixedArray $keyCache;
    public SplFixedArray $valueCache;

    public function __construct(Config $config)
    {
        $this->x = new SplFixedArray($config->dim);
        $this->xb = new SplFixedArray($config->dim);
        $this->xb2 = new SplFixedArray($config->dim);
        $this->hb = new SplFixedArray($config->hiddenDim);
        $this->hb2 = new SplFixedArray($config->hiddenDim);
        $this->q = new SplFixedArray($config->dim);
        $this->k = new SplFixedArray($config->dim);
        $this->v = new SplFixedArray($config->dim);
        $this->att = new SplFixedArray($config->nHeads * $config->seqLen);
        $this->logits = new SplFixedArray($config->vocabSize);
        $this->keyCache = new SplFixedArray($config->nLayers * $config->seqLen * $config->dim);
        $this->valueCache = new SplFixedArray($config->nLayers * $config->seqLen * $config->dim);
    }
}

function rmsNorm(SplFixedArray $output, SplFixedArray $input, SplFixedArray $weight, int $size, int $weightOffset = 0): void
{
    $sumOfSquares = 0.0;
    for ($index = 0; $index < $size; $index++) {
        $sumOfSquares += $input[$index] * $input[$index];
    }
    $sumOfSquares /= $size;
    $sumOfSquares += 1e-5;
    $scale = 1.0 / sqrt($sumOfSquares);
    for ($index = 0; $index < $size; $index++) {
        $output[$index] = $weight[$weightOffset + $index] * ($scale * $input[$index]);
    }
}

function softmax(SplFixedArray $array, int $size, int $offset = 0): void
{
    $maxValue = $array[$offset];
    for ($index = 1; $index < $size; $index++) {
        if ($array[$offset + $index] > $maxValue) {
            $maxValue = $array[$offset + $index];
        }
    }
    $sum = 0.0;
    for ($index = 0; $index < $size; $index++) {
        $array[$offset + $index] = exp($array[$offset + $index] - $maxValue);
        $sum += $array[$offset + $index];
    }
    for ($index = 0; $index < $size; $index++) {
        $array[$offset + $index] /= $sum;
    }
}

function matMul(SplFixedArray $output, SplFixedArray $input, SplFixedArray $weights, int $inputDim, int $outputDim, int $weightOffset = 0): void
{
    for ($outIndex = 0; $outIndex < $outputDim; $outIndex++) {
        $value = 0.0;
        $rowWeightOffset = $outIndex * $inputDim + $weightOffset;
        for ($inIndex = 0; $inIndex < $inputDim; $inIndex++) {
            $value += $weights[$rowWeightOffset + $inIndex] * $input[$inIndex];
        }
        $output[$outIndex] = $value;
    }
}

function transformer(int $token, int $pos, Config $config, RunState $state, TransformerWeights $weights): SplFixedArray
{
    $dim = $config->dim;
    $hiddenDim = $config->hiddenDim;
    $headSize = (int)($dim / $config->nHeads);
    $kvDim = (int)($config->dim * $config->nKvHeads / $config->nHeads);
    $kvMul = (int)($config->nHeads / $config->nKvHeads);

    // Embedding lookup
    for ($i = 0; $i < $dim; $i++) {
        $state->x[$i] = $weights->tokenEmbeddingTable[$token * $dim + $i];
    }

    for ($layer = 0; $layer < $config->nLayers; $layer++) {
        // Attention RMSNorm
        rmsNorm($state->xb, $state->x, $weights->rmsAttWeight, $dim, $layer * $dim);

        // QKV matmuls
        matMul($state->q, $state->xb, $weights->wq, $dim, $dim, $layer * $dim * $dim);
        matMul($state->k, $state->xb, $weights->wk, $dim, $kvDim, $layer * $dim * $kvDim);
        matMul($state->v, $state->xb, $weights->wv, $dim, $kvDim, $layer * $dim * $kvDim);

        // RoPE relative positional encoding
        for ($i = 0; $i < $dim; $i += 2) {
            $headDim = $i % $headSize;
            $freq = 1.0 / pow(10000.0, $headDim / $headSize);
            $val = $pos * $freq;
            $fcr = cos($val);
            $fci = sin($val);
            $rotn = ($i < $kvDim) ? 2 : 1;
            for ($v = 0; $v < $rotn; $v++) {
                $vec = ($v == 0) ? $state->q : $state->k;
                $v0 = $vec[$i];
                $v1 = $vec[$i + 1];
                $vec[$i] = $v0 * $fcr - $v1 * $fci;
                $vec[$i + 1] = $v0 * $fci + $v1 * $fcr;
            }
        }

        // KV cache
        $layerOffset = $layer * $config->seqLen * $kvDim;
        for ($i = 0; $i < $kvDim; $i++) {
            $state->keyCache[$layerOffset + $pos * $kvDim + $i] = $state->k[$i];
            $state->valueCache[$layerOffset + $pos * $kvDim + $i] = $state->v[$i];
        }

        // Multihead attention
        for ($head = 0; $head < $config->nHeads; $head++) {
            $qOffset = $head * $headSize;
            $attOffset = $head * $config->seqLen;

            for ($t = 0; $t <= $pos; $t++) {
                $kOffset = $layerOffset + $t * $kvDim + (int)($head / $kvMul) * $headSize;
                $score = 0.0;
                for ($i = 0; $i < $headSize; $i++) {
                    $score += $state->q[$qOffset + $i] * $state->keyCache[$kOffset + $i];
                }
                $score /= sqrt($headSize);
                $state->att[$attOffset + $t] = $score;
            }

            softmax($state->att, $pos + 1, $attOffset);

            $xbOffset = $head * $headSize;
            for ($i = 0; $i < $headSize; $i++) {
                $state->xb[$xbOffset + $i] = 0.0;
            }

            for ($t = 0; $t <= $pos; $t++) {
                $vOffset = $layerOffset + $t * $kvDim + (int)($head / $kvMul) * $headSize;
                $a = $state->att[$attOffset + $t];
                for ($i = 0; $i < $headSize; $i++) {
                    $state->xb[$xbOffset + $i] += $a * $state->valueCache[$vOffset + $i];
                }
            }
        }

        // Final attention matmul
        matMul($state->xb2, $state->xb, $weights->wo, $dim, $dim, $layer * $dim * $dim);

        // Residual connection
        for ($i = 0; $i < $dim; $i++) {
            $state->x[$i] += $state->xb2[$i];
        }

        // FFN RMSNorm
        rmsNorm($state->xb, $state->x, $weights->rmsFfnWeight, $dim, $layer * $dim);

        // FFN
        matMul($state->hb, $state->xb, $weights->w1, $dim, $hiddenDim, $layer * $dim * $hiddenDim);
        matMul($state->hb2, $state->xb, $weights->w3, $dim, $hiddenDim, $layer * $dim * $hiddenDim);

        // SwiGLU non-linearity
        for ($i = 0; $i < $hiddenDim; $i++) {
            $val = $state->hb[$i];
            $val *= (1.0 / (1.0 + exp(-$val))); // SiLU
            $val *= $state->hb2[$i];
            $state->hb[$i] = $val;
        }

        // Final FFN matmul
        matMul($state->xb, $state->hb, $weights->w2, $hiddenDim, $dim, $layer * $dim * $hiddenDim);

        // Residual connection
        for ($i = 0; $i < $dim; $i++) {
            $state->x[$i] += $state->xb[$i];
        }
    }

    // Final RMSNorm
    rmsNorm($state->x, $state->x, $weights->rmsFinalWeight, $dim);

    // Classifier
    matMul($state->logits, $state->x, $weights->wcls, $dim, $config->vocabSize);

    return $state->logits;
}

class Tokenizer
{
    public array $vocab;
    public SplFixedArray $vocabScores;
    public int $vocabSize;
    public int $maxTokenLength;

    public function __construct(string $path, int $vocabSize)
    {
        $this->vocabSize = $vocabSize;
        $fileHandle = fopen($path, "rb");
        $this->maxTokenLength = unpack("i", fread($fileHandle, 4))[1];
        $this->vocab = [];
        $this->vocabScores = new SplFixedArray($vocabSize);
        for ($i = 0; $i < $vocabSize; $i++) {
            $this->vocabScores[$i] = unpack("f", fread($fileHandle, 4))[1];
            $len = unpack("i", fread($fileHandle, 4))[1];
            $this->vocab[$i] = fread($fileHandle, $len);
        }
        fclose($fileHandle);
    }

    public function decode(int $prevToken, int $token): string
    {
        $piece = $this->vocab[$token];
        if ($prevToken == 1 && $piece[0] == ' ') {
            $piece = substr($piece, 1);
        }
        if (preg_match('/<0x([0-9A-F]{2})>/i', $piece, $matches)) {
            return chr((int)hexdec($matches[1]));
        }
        return $piece;
    }

    public function encode(string $text, bool $bos, bool $eos): array
    {
        $tokens = [];
        if ($bos) {
            $tokens[] = 1;
        }

        // Karpathy's tokenizer is BPE, this is a simplified greedy version
        $str = $text;
        if ($str !== "") {
            // Prepend dummy space if not empty
            $str = " " . $str;
        }

        $i = 0;
        while ($i < strlen($str)) {
            $bestLen = -1;
            $bestToken = -1;

            for ($t = 0; $t < $this->vocabSize; $t++) {
                $v = $this->vocab[$t];
                $len = strlen($v);
                if ($len > $bestLen && substr($str, $i, $len) === $v) {
                    $bestLen = $len;
                    $bestToken = $t;
                }
            }

            if ($bestToken !== -1) {
                $tokens[] = $bestToken;
                $i += $bestLen;
            } else {
                // Just skip unknown byte
                $i++;
            }
        }

        if ($eos) {
            $tokens[] = 2;
        }
        return $tokens;
    }
}

function sample(SplFixedArray $logits, int $vocabSize, float $temperature = 1.0): int
{
    if ($temperature == 0.0) {
        // Greedy argmax
        $maxIndex = 0;
        $maxProb = $logits[0];
        for ($i = 1; $i < $vocabSize; $i++) {
            if ($logits[$i] > $maxProb) {
                $maxProb = $logits[$i];
                $maxIndex = $i;
            }
        }
        return $maxIndex;
    } else {
        // Temperature sampling
        for ($i = 0; $i < $vocabSize; $i++) {
            $logits[$i] /= $temperature;
        }
        softmax($logits, $vocabSize);
        $randomValue = (float)mt_rand() / (float)mt_getrandmax();
        $cumulativeProbability = 0.0;
        for ($i = 0; $i < $vocabSize; $i++) {
            $cumulativeProbability += $logits[$i];
            if ($randomValue < $cumulativeProbability) {
                return $i;
            }
        }
        return $vocabSize - 1;
    }
}

// ********
// * Main *
// ********

if ($argc < 2) {
    echo "Usage: php run.php <checkpoint.bin> [temperature] [steps] [prompt]\n";
    exit(1);
}

$checkpointPath = $argv[1];
$temperature = (float)($argv[2] ?? 0.9);
$steps = (int)($argv[3] ?? 256);
$prompt = $argv[4] ?? "";

$startLoading = hrtime(true);
$checkpointFile = fopen($checkpointPath, "rb");
$config = new Config($checkpointFile);
$weights = new TransformerWeights($config, $checkpointFile);
fclose($checkpointFile);
$endLoading = hrtime(true);

$state = new RunState($config);
if (!file_exists("tokenizer.bin")) {
    echo "Error: tokenizer.bin not found\n";
    exit(1);
}
$tokenizer = new Tokenizer("tokenizer.bin", $config->vocabSize);

$promptTokens = $prompt !== "" ? $tokenizer->encode($prompt, true, false) : [1];
$numPromptTokens = count($promptTokens);

$pos = 0;
$token = $promptTokens[0];

$startGen = hrtime(true);
$tokensGenerated = 0;

while ($pos < $steps) {
    $logits = transformer($token, $pos, $config, $state, $weights);

    if ($pos < $numPromptTokens - 1) {
        $next = $promptTokens[$pos + 1];
    } else {
        $next = sample($logits, $config->vocabSize, $temperature);
    }

    $piece = $tokenizer->decode($token, $next);
    echo $piece;
    flush();

    $token = $next;
    $pos++;
    $tokensGenerated++;

    if ($token == 2) {
        break;
    }
}
$endGen = hrtime(true);

$loadingTime = ($endLoading - $startLoading) / 1e9;
$genTime = ($endGen - $startGen) / 1e9;
$tps = $tokensGenerated / $genTime;
$memory = memory_get_peak_usage(true) / 1024 / 1024;

echo "\n\n";
echo "--- PHP Llama2 Benchmark ---\n";
echo sprintf("Model Load Time : %.3f s\n", $loadingTime);
echo sprintf("Generation Time : %.3f s\n", $genTime);
echo sprintf("Tokens Processed: %d\n", $tokensGenerated);
echo sprintf("Throughput      : %.2f tok/s\n", $tps);
echo sprintf("Peak Memory     : %.2f MB\n", $memory);
echo "----------------------------\n";
