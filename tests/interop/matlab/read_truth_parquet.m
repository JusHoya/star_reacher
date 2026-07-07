% read_truth_parquet.m -- MATLAB parquetread interop check (Phase 5 exit
% criterion 3, D-15) for the Parquet files written by `star export --parquet`
% from the missions/twobody_leo.toml run (layout: docs/formats/parquet_v1.md).
%
% Inputs: truth.parquet and events.parquet in the directory named by the
% STAR_REACHER_EXPORT_DIR environment variable (default: the current
% directory). Regeneration procedure: README.md; provenance and the
% derivation of every expected constant below: manifest.toml.
%
% Numeric comparisons are bit-exact through num2hex: the Parquet contract is
% that every stored IEEE-754 double survives unreformatted
% (docs/formats/parquet_v1.md section 3), so a tolerance would hide exactly
% the type or round-trip corruption this item exists to catch. The first
% truth row is the mission's initial state exactly as written in
% missions/twobody_leo.toml; the last row was captured from the generating
% run pinned in manifest.toml.

srdir = getenv('STAR_REACHER_EXPORT_DIR');
if isempty(srdir)
    srdir = pwd;
end
fprintf('MATLAB version: %s\n', version);
truthPath = fullfile(srdir, 'truth.parquet');
eventsPath = fullfile(srdir, 'events.parquet');
print_sha256(truthPath);
print_sha256(eventsPath);

results = true(0, 1);

% --- truth.parquet -------------------------------------------------------
T = parquetread(truthPath);
results(end+1) = check(height(T) == 54001, ...
    sprintf('truth rows = %d (expect 54001 = 5400 s at 10 Hz plus t = 0)', height(T)));

expectedCols = {'t_s', 'r_m_0', 'r_m_1', 'r_m_2', 'v_mps_0', 'v_mps_1', ...
    'v_mps_2', 'q_i2b_0', 'q_i2b_1', 'q_i2b_2', 'q_i2b_3', ...
    'w_b_radps_0', 'w_b_radps_1', 'w_b_radps_2', 'mass_kg'};
results(end+1) = check(isequal(T.Properties.VariableNames, expectedCols), ...
    'truth column names and order match the flattened channel list');
results(end+1) = check(all(varfun(@(x) isa(x, 'double'), T, ...
    'OutputFormat', 'uniform')), 'every truth column is double');

% First row: the mission initial state (GCRF), identity attitude, zero
% rates, 150 kg -- the exact literals of missions/twobody_leo.toml.
firstHex = {'0000000000000000', '4159db4640000000', '0000000000000000', ...
    '0000000000000000', '0000000000000000', '40bdf4999999999a', ...
    '0000000000000000', '3ff0000000000000', '0000000000000000', ...
    '0000000000000000', '0000000000000000', '0000000000000000', ...
    '0000000000000000', '0000000000000000', '4062c00000000000'};
% Last row (t = 5400 s), captured from the generating run.
lastHex = {'40b5180000000000', '4159776df996aa71', 'c131e56f1155a2b5', ...
    '0000000000000000', '4094bba682683f0e', '40bd80ed5e14389f', ...
    '0000000000000000', '3ff0000000000000', '0000000000000000', ...
    '0000000000000000', '0000000000000000', '0000000000000000', ...
    '0000000000000000', '0000000000000000', '4062c00000000000'};
results(end+1) = check(row_matches(T, expectedCols, 1, firstHex), ...
    'first truth row bit-exact (mission initial state)');
results(end+1) = check(row_matches(T, expectedCols, height(T), lastHex), ...
    'last truth row bit-exact (t = 5400 s state from the pinned run)');

% --- events.parquet ------------------------------------------------------
E = parquetread(eventsPath);
results(end+1) = check(height(E) == 2, ...
    sprintf('events rows = %d (expect 2: run_start, run_end)', height(E)));
results(end+1) = check(isequal(E.Properties.VariableNames, ...
    {'t_s', 'code', 'detail'}), 'events column names and order');
results(end+1) = check(isa(E.t_s, 'double') && isa(E.code, 'uint32'), ...
    'events t_s is double and code survives as uint32 (not widened/signed)');
fprintf('     (events.detail read back as class %s)\n', class(E.detail));
results(end+1) = check(isequal(E.code, uint32([1; 2])), 'events codes are uint32 1, 2');
results(end+1) = check(isequal(string(E.detail), ["run_start"; "run_end"]), ...
    'events detail strings are run_start, run_end');
results(end+1) = check(strcmp(num2hex(E.t_s(1)), '0000000000000000') && ...
    strcmp(num2hex(E.t_s(2)), '40b5180000000000'), ...
    'events epochs bit-exact (0 s and 5400 s)');

% --- verdict --------------------------------------------------------------
if all(results)
    fprintf('MATLAB-PARQUET: PASS (%d/%d)\n', numel(results), numel(results));
else
    fprintf('MATLAB-PARQUET: FAIL (%d/%d)\n', nnz(results), numel(results));
    error('read_truth_parquet:failed', '%d check(s) failed', nnz(~results));
end

% --- local functions ------------------------------------------------------
function ok = check(cond, desc)
ok = logical(cond);
if ok
    fprintf('ok   %s\n', desc);
else
    fprintf('FAIL %s\n', desc);
end
end

function ok = row_matches(T, cols, rowIndex, expectedHex)
ok = true;
for k = 1:numel(cols)
    % Guard so a missing column reports FAIL instead of aborting the run.
    if ~ismember(cols{k}, T.Properties.VariableNames)
        fprintf('     missing column %s\n', cols{k});
        ok = false;
        continue
    end
    got = num2hex(T.(cols{k})(rowIndex));
    if ~strcmp(got, expectedHex{k})
        fprintf('     mismatch %s row %d: got %s expect %s\n', ...
            cols{k}, rowIndex, got, expectedHex{k});
        ok = false;
    end
end
end

function print_sha256(path)
% Input provenance for the committed transcript. Desktop MATLAB carries a
% JVM; under matlab -nojvm, hash the files externally instead.
if usejava('jvm')
    fid = fopen(path, 'r');
    if fid < 0
        fprintf('sha256 unavailable (cannot open)  %s\n', path);
        return
    end
    bytes = fread(fid, inf, 'uint8=>uint8');
    fclose(fid);
    md = java.security.MessageDigest.getInstance('SHA-256');
    md.update(bytes);
    d = typecast(md.digest(), 'uint8');
    fprintf('sha256 %s  %s\n', lower(reshape(dec2hex(d, 2).', 1, [])), path);
else
    fprintf('sha256 unavailable (no JVM); hash %s externally\n', path);
end
end
