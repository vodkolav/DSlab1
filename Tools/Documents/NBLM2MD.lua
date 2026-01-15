-- Lua filter for pandoc to process HTML content based on user requirements

function Div(el)

--   -- Ignore placeholder comments (<!---->)
--   if #el.content == 0 or pandoc.utils.stringify(el.content):match("^%s*$") then
--     return {"bruh"}
--   end

  if el.classes:includes("paragraph") then
    -- Extract attributes and content
    -- local attributes = el.c[1]  -- The first element contains the attributes
    -- local content = {table.unpack(el.c, 2)}  -- The rest are the content blocks

    -- Collect all inline content from Plain blocks
    local merged_content = {}
    for _, block in ipairs(el.c) do
        -- if block.t == "Plain" then
            for _, inline in ipairs(block.c) do
                -- table.insert(merged_content, pandoc.Space())  -- Add space between inlines from different Plain blocks
                table.insert(merged_content, inline)
            end
        -- else
        -- If there's a non-Plain block, keep the Div as-is
        -- return el
        -- end
    end

  -- Replace the Div's content with a single Plain block
    return pandoc.Div(pandoc.Plain(merged_content), el.attr)
  end


  -- Requirement 1: Only transform inner content of div with class="chat-panel-content"
  if el.classes:includes("chat-panel-content") then
    return el.content
  end

  -- Requirement 2: Ignore all immediate children divs of class="chat-panel-empty-state ng-star-inserted"
  if el.classes:includes("chat-panel-empty-state") and el.classes:includes("ng-star-inserted") then
    for i = #el.content, 1, -1 do
      if el.content[i].t == "Div" then
        table.remove(el.content, i)
      end
    end
    return el
  end

  -- Requirement 3: Add a vertical line (---) before inner content of div with class="chat-message-pair ng-star-inserted"
  if el.classes:includes("chat-message-pair") and el.classes:includes("ng-star-inserted") then
    return pandoc.Blocks({pandoc.HorizontalRule(), el})
  end

  -- Requirement 4: Add "**USER:**  " before inner content of div with class="from-user-container"
  if el.classes:includes("from-user-container") then
    table.insert(el.content, 1, pandoc.Strong("USER:"))
    table.insert(el.content, 2, pandoc.Space())
    table.insert(el.content, 3, pandoc.Space())
    return el
  end

  -- Requirement 5: Add "**CLANKER:**  " before inner content of div with class="to-user-container"
  if el.classes:includes("to-user-container") then
    table.insert(el.content, 1, pandoc.Strong("CLANKER:"))
    table.insert(el.content, 2, pandoc.Space())
    table.insert(el.content, 3, pandoc.Space())
    return el
  end

  -- Requirement 6: Ignore <mat-card-actions> tags
  if el.t == "mat-card-actions" then
    return {}
  end

  -- Requirement 7: Ignore divs of class="suggestions-container ng-star-inserted"
  if el.classes:includes("suggestions-container") and el.classes:includes("ng-star-inserted") then
    return {}
  end

  return el
end


function Span(el)
  -- Requirement 8: Convert <span> with aria-label="label" into wiki links [[label|content]]
  if el.attributes["aria-label"] then
    local label = el.attributes["aria-label"]
    local content = pandoc.utils.stringify(el.content)

    -- Remove "x: " at the beginning of the label
    label = label:gsub("^%d+: ", "")

    -- Replace ":", "\", and "/" with "-"
    label = label:gsub("[:\\/]", "-")

    return pandoc.RawInline("markdown", string.format(" [[%s|%s]]", label, content))

  end

  --Flatten nested inline elements (e.g., <button> inside <span>)
  -- el.content = flattenInlineElements(el.content)

  --Ensure inline spans are merged properly
  return el
end

-- function Button(el)
--   -- Treat <button> tags as inline elements by returning their content
--   return flattenInlineElements(el.content)
-- end

-- local function starts_with(start, str)
--   return str:sub(1, #start) == start
-- end

-- function Plain(el)
--     -- appr 1
--   -- Ignore placeholder comments (<!---->)
-- --   if #el.c > 0 then -- or pandoc.utils.stringify(el.content):match("^%s*$") then
--     return el.c.t
-- --   end
-- --  return el
--     -- return pandoc.Span(el.c)
-- end

-- function Plain(el)
--     -- appr 2
--   -- Ignore placeholder comments (<!---->)
-- --   if #el.c > 0 then -- or pandoc.utils.stringify(el.content):match("^%s*$") then
--     -- return el.c.t
-- --   end
-- --  return el
--     -- el.text = "Span"
--     -- return el
--     return el.content
-- end

function RawInline(el)
  -- Ignore HTML comments (<!---->)

    if el.text and starts_with('<!--', el.text) then
        return {}
    end

  --   if el.text:match("^<!%-%-.*%-%->$") then
--     return {"bruh"}
--   end
  return el
end

-- function RawBlock(el)
--   -- Ignore HTML comments (<!---->) in block context
--   if  el.text:match("^<!%-%-.*%-%->$") then
--     return {}
--   end
--   return el
-- end


function flattenInlineElements(content)
  -- Helper function to flatten nested inline elements and remove placeholders
  local flattened = {}
  for _, elem in ipairs(content) do
    if elem.t == "RawInline" and elem.format == "html" and elem.text:match("^<!---->$") then
      -- Ignore placeholder comments
    elseif elem.t == "Span" or elem.t == "Button" then
      -- Recursively flatten nested inline elements
      local innerContent = flattenInlineElements(elem.content)
      for _, innerElem in ipairs(innerContent) do
        table.insert(flattened, innerElem)
      end
    else
      table.insert(flattened, elem)
    end
  end
  return flattened
end

function Inlines(inlines)
  -- Merge consecutive inline elements to prevent extra newlines
  local merged = {}
  for _, inline in ipairs(inlines) do
    if inline.t == "Str" and #merged > 0 and merged[#merged].t == "Str" then
      merged[#merged].text = merged[#merged].text .. inline.text
    else
      table.insert(merged, inline)
    end
  end
  return merged
end